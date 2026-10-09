"""Direct 2-bit affine matvec and normalized Hadamard kernels (NVRTC, no C++ build)."""

from __future__ import annotations
import os
from pathlib import Path
import torch

# CuPy needs the CUDA toolkit's NVRTC DLL. Prefer CUDA 12.8 when several toolkits exist.
if os.name == "nt":
    toolkit = Path(
        os.environ.get(
            "CUDA_PATH_V12_8",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        )
    )
    if toolkit.is_dir():
        os.environ["CUDA_PATH"] = str(toolkit)
        _dll_dir = os.add_dll_directory(str(toolkit / "bin"))
import cupy as cp

_SOURCE = r"""
#include <cuda_fp16.h>
#include <mma.h>
#ifndef TILE_STRIDE
#define TILE_STRIDE 128
#endif
extern "C" __global__ void rotate(const half* x, const half* signs, half* y,
                                  int width, int block, int inverse) {
    extern __shared__ float v[];
    int base = blockIdx.x * block;
    for (int j=threadIdx.x; j<block; j+=blockDim.x)
        v[j] = __half2float(x[base+j]) * (inverse ? 1.f : __half2float(signs[(base+j)%width]));
    __syncthreads();
    for (int stride=1; stride<block; stride*=2) {
        for (int j=threadIdx.x; j<block/2; j+=blockDim.x) {
            int a=(j/stride)*2*stride+j%stride, b=a+stride;
            float va=v[a], vb=v[b]; v[a]=va+vb; v[b]=va-vb;
        }
        __syncthreads();
    }
    for (int j=threadIdx.x; j<block; j+=blockDim.x)
        y[base+j] = __float2half(v[j] * rsqrtf((float)block) * (inverse ? __half2float(signs[(base+j)%width]) : 1.f));
}
extern "C" __global__ void packed_mm(const half* x, const unsigned int* w,
                                     const half* s, const half* b, half* y,
                                     int rows, int width, int tokens, int ternary) {
    int row=blockIdx.x, token=blockIdx.y;
    float sum=0.f;
    for (int word=threadIdx.x; word<width/16; word+=blockDim.x) {
        unsigned int bits=w[row*(width/16)+word];
        int group=row*(width/128)+word/8;
        float scale=__half2float(s[group]), bias=ternary ? -scale : __half2float(b[group]);
        #pragma unroll
        for (int j=0; j<16; ++j) {
            // MLX affine dequantization rounds back to fp16 before the dot product.
            half val=__float2half(((bits>>(2*j))&3)*scale+bias);
            sum += __half2float(x[token*width+word*16+j])*__half2float(val);
        }
    }
    for (int offset=16; offset>0; offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    __shared__ float partial[8];
    if (threadIdx.x%32==0) partial[threadIdx.x/32]=sum;
    __syncthreads();
    if (threadIdx.x<32) {
        sum=threadIdx.x<8 ? partial[threadIdx.x] : 0.f;
        for (int offset=16; offset>0; offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if (threadIdx.x==0) y[token*rows+row]=__float2half(sum);
    }
}
// Four independent warps avoid block reductions and idle lanes in small-K GEMV.
extern "C" __global__ void ternary_mv(const half* x, const unsigned int* w,
                                    const half* s, half* y, int rows, int width, int tokens) {
    int lane=threadIdx.x%32, row=blockIdx.x*4+threadIdx.x/32, token=blockIdx.y;
    if (row>=rows) return;
    float sum=0.f;
    for (int word=lane; word<width/16; word+=32) {
        unsigned int bits=w[row*(width/16)+word];
        float scale=__half2float(s[row*(width/128)+word/8]);
        #pragma unroll
        for (int j=0; j<16; ++j)
            sum += __half2float(x[token*width+word*16+j]) * (float(int((bits>>(2*j))&3)-1)*scale);
    }
    for (int offset=16; offset>0; offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if (lane==0) y[token*rows+row]=__float2half(sum);
}
// Packed ternary weights are unpacked only into a bounded 16x64x128 shared tile.
// Tensor cores reuse that tile across 16 prompt tokens; no dense matrix is retained.
extern "C" __global__ void ternary_mma(const half* x, const unsigned int* w,
                                     const half* s, half* y, int rows, int width, int tokens) {
    using namespace nvcuda;
    __shared__ __align__(32) half a[16*TILE_STRIDE];
    __shared__ __align__(32) half b[64*TILE_STRIDE];
    __shared__ __align__(32) float c[16*64];
    int warp=threadIdx.x/32, r0=blockIdx.x*64, t0=blockIdx.y*16;
    wmma::fragment<wmma::accumulator,16,16,16,float> accum;
    wmma::fill_fragment(accum,0.f);
    for (int k0=0; k0<width; k0+=128) {
        for (int i=threadIdx.x; i<16*128; i+=blockDim.x) {
            int token=t0+i/128;
            a[(i/128)*TILE_STRIDE+i%128]=token<tokens ? x[token*width+k0+i%128] : __float2half(0.f);
        }
        for (int i=threadIdx.x; i<64*8; i+=blockDim.x) {
            int row=r0+i/8, word=k0/16+i%8;
            unsigned int bits=row<rows ? w[row*(width/16)+word] : 0x55555555u;
            float scale=row<rows ? __half2float(s[row*(width/128)+k0/128]) : 0.f;
            #pragma unroll
            for (int j=0; j<16; ++j)
                b[(i/8)*TILE_STRIDE+(i%8)*16+j]=__float2half(float(int((bits>>(2*j))&3)-1)*scale);
        }
        __syncthreads();
        #pragma unroll
        for (int k=0; k<128; k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,a+k,TILE_STRIDE);
            wmma::load_matrix_sync(bf,b+warp*16*TILE_STRIDE+k,TILE_STRIDE);
            wmma::mma_sync(accum,af,bf,accum);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c+warp*16,accum,64,wmma::mem_row_major);
    __syncthreads();
    for (int i=threadIdx.x; i<16*64; i+=blockDim.x) {
        int token=t0+i/64, row=r0+i%64;
        if (token<tokens && row<rows) y[token*rows+row]=__float2half(c[i]);
    }
}
"""
_rotate = cp.RawKernel(_SOURCE, "rotate", options=("--std=c++11",))
_mm = cp.RawKernel(_SOURCE, "packed_mm", options=("--std=c++11",))
_mv = cp.RawKernel(_SOURCE, "ternary_mv", options=("--std=c++11",))
_mma = cp.RawKernel(_SOURCE, "ternary_mma", options=("--std=c++11",))
_mma_padded = cp.RawKernel(
    _SOURCE, "ternary_mma", options=("--std=c++11", "-DTILE_STRIDE=136")
)


def _array(t):
    return cp.from_dlpack(t.detach() if t.requires_grad else t)


def rotate(x, block, signs, inverse=False):
    x = x.to(torch.float16).contiguous()
    out = torch.empty_like(x)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _rotate(
            (x.numel() // block,),
            (256,),
            (
                _array(x),
                _array(signs),
                _array(out),
                cp.int32(x.shape[-1]),
                cp.int32(block),
                cp.int32(inverse),
            ),
            shared_mem=block * 4,
        )
    return out


def matmul(x, weight, scales, biases):
    x = x.to(torch.float16).contiguous()
    rows, width = weight.shape[0], weight.shape[1] * 16
    out = torch.empty((*x.shape[:-1], rows), device=x.device, dtype=x.dtype)
    tokens = x.numel() // width
    if biases is None and os.environ.get("BONSAI_CUDA_LEGACY_MM") != "1":
        kernel = (
            (
                _mma_padded
                if os.environ.get("BONSAI_CUDA_MMA_PADDING", "1") == "1"
                else _mma
            )
            if tokens >= 8
            else _mv
        )
        grid = (
            ((rows + 63) // 64, (tokens + 15) // 16)
            if tokens >= 8
            else ((rows + 3) // 4, tokens)
        )
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            kernel(
                grid,
                (128,),
                (
                    _array(x),
                    _array(weight),
                    _array(scales),
                    _array(out),
                    cp.int32(rows),
                    cp.int32(width),
                    cp.int32(tokens),
                ),
            )
        return out
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _mm(
            (rows, x.numel() // width),
            (256,),
            (
                _array(x),
                _array(weight),
                _array(scales),
                _array(scales if biases is None else biases),
                _array(out),
                cp.int32(rows),
                cp.int32(width),
                cp.int32(x.numel() // width),
                cp.int32(biases is None),
            ),
        )
    return out
