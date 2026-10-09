"""Two-kernel cached GDN update, matching the portable HF recurrence and gated norm."""

import torch
from .cuda_kernels import cp, _array

_SOURCE = r"""
#include <cuda_fp16.h>
extern "C" __global__ void conv_update(const half* x, half* state, const half* w,
                                     half* out, int channels, int history, int batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=batch*channels) return;
    int channel=i%channels;
    float value=0.f;
    for (int k=0; k<history-1; ++k) {
        half next=state[i*history+k+1];
        value+=__half2float(next)*__half2float(w[channel*history+k]);
        state[i*history+k]=next;
    }
    half next=x[i];
    state[i*history+history-1]=next;
    value+=__half2float(next)*__half2float(w[channel*history+history-1]);
    // Grouped fp16 convolution and SiLU each round to fp16 in the reference.
    value=__half2float(__float2half(value));
    out[i]=__float2half(value/(1.f+expf(-value)));
}
extern "C" __global__ void recurrent_norm(const half* mixed, const half* z,
    const half* a, const half* b, const half* alog, const half* dt, const half* norm,
    half* state, half* out, int kh, int vh, int kd, int vd, float normeps) {
    int head=blockIdx.x%vh, batch=blockIdx.x/vh, j=threadIdx.x;
    int qhead=head/(vh/kh), channels=2*kh*kd+vh*vd;
    __shared__ float q[256], k[256], tmpq[8], tmpk[8], tmpo[8];
    __shared__ float decay, beta, inv;
    float qr=j<kd ? __half2float(mixed[batch*channels+qhead*kd+j]) : 0.f;
    float kr=j<kd ? __half2float(mixed[batch*channels+kh*kd+qhead*kd+j]) : 0.f;
    float qs=__half2float(__float2half(qr*qr)), ks=__half2float(__float2half(kr*kr));
    for(int d=16;d>0;d/=2) {qs+=__shfl_down_sync(0xffffffff,qs,d);ks+=__shfl_down_sync(0xffffffff,ks,d);}
    if(j%32==0) {tmpq[j/32]=qs;tmpk[j/32]=ks;}
    __syncthreads();
    if(j==0) {
        qs=ks=0.f;
        for(int i=0;i<blockDim.x/32;++i) {qs+=tmpq[i];ks+=tmpk[i];}
        // l2norm in HF operates in the activation dtype before recurrence casts to fp32.
        qs=__half2float(__hadd(__float2half(qs),__float2half(1e-6f)));
        ks=__half2float(__hadd(__float2half(ks),__float2half(1e-6f)));
        tmpq[0]=__half2float(__float2half(rsqrtf(qs)));
        tmpk[0]=__half2float(__float2half(rsqrtf(ks)));
        float av=__half2float(a[batch*vh+head])+__half2float(dt[head]);
        float soft=av>20.f ? av : log1pf(expf(av));
        decay=expf(-expf(__half2float(alog[head]))*soft);
        beta=__half2float(__float2half(1.f/(1.f+expf(-__half2float(b[batch*vh+head])))));
    }
    __syncthreads();
    if(j<kd) {
        q[j]=__half2float(__float2half(qr*tmpq[0]))*rsqrtf(float(kd));
        k[j]=__half2float(__float2half(kr*tmpk[0]));
    }
    __syncthreads();
    float result=0.f;
    if(j<vd) {
        int base=(batch*vh+head)*kd*vd;
        float mem=0.f;
        for(int i=0;i<kd;++i) {
            float old=__half2float(state[base+i*vd+j])*decay;
            mem+=old*k[i];
        }
        float value=__half2float(mixed[batch*channels+2*kh*kd+head*vd+j]);
        float delta=(value-mem)*beta;
        for(int i=0;i<kd;++i) {
            float next=__half2float(state[base+i*vd+j])*decay+k[i]*delta;
            state[base+i*vd+j]=__float2half(next);
            result+=next*q[i];
        }
        result=__half2float(__float2half(result));
    }
    float square=result*result;
    for(int d=16;d>0;d/=2) square+=__shfl_down_sync(0xffffffff,square,d);
    if(j%32==0) tmpo[j/32]=square;
    __syncthreads();
    if(j==0) {
        square=0.f;
        for(int i=0;i<blockDim.x/32;++i) square+=tmpo[i];
        inv=rsqrtf(square/float(vd)+normeps);
    }
    __syncthreads();
    if(j<vd) {
        half normalized=__float2half(result*inv);
        half weighted=__hmul(normalized,norm[j]);
        int index=(batch*vh+head)*vd+j;
        float gate=__half2float(z[index]);
        out[index]=__float2half(__half2float(weighted)*gate/(1.f+expf(-gate)));
    }
}
"""
# Preserve separate fp32 multiply/add operations from the reference recurrence.
_conv = cp.RawKernel(_SOURCE, "conv_update", options=("--std=c++11", "--fmad=false"))
_recur = cp.RawKernel(
    _SOURCE, "recurrent_norm", options=("--std=c++11", "--fmad=false")
)


def cached_update(layer, mixed, z, a, b, cache):
    states = cache.layers[layer.layer_idx]
    conv = states.conv_states.contiguous()
    states.conv_states = conv
    # Accumulate in fp32 registers and round only the cached state back to fp16,
    # exactly where HF copies its fp32 recurrence result into the fp16 cache.
    state = states.recurrent_states.contiguous()
    states.recurrent_states = state
    batch, channels, _ = mixed.shape
    filtered = torch.empty((batch, channels), device=mixed.device, dtype=mixed.dtype)
    out = torch.empty_like(z)
    threads = max(32, ((max(layer.head_k_dim, layer.head_v_dim) + 31) // 32) * 32)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _conv(
            ((batch * channels + 255) // 256,),
            (256,),
            (
                _array(mixed.contiguous()),
                _array(conv),
                _array(layer.conv1d.weight),
                _array(filtered),
                cp.int32(channels),
                cp.int32(layer.conv_kernel_size),
                cp.int32(batch),
            ),
        )
        _recur(
            (batch * layer.num_v_heads,),
            (threads,),
            (
                _array(filtered),
                _array(z),
                _array(a),
                _array(b),
                _array(layer.A_log),
                _array(layer.dt_bias),
                _array(layer.norm.weight),
                _array(state),
                _array(out),
                cp.int32(layer.num_k_heads),
                cp.int32(layer.num_v_heads),
                cp.int32(layer.head_k_dim),
                cp.int32(layer.head_v_dim),
                cp.float32(layer.norm.variance_epsilon),
            ),
        )
    return out.reshape(batch, 1, -1)
