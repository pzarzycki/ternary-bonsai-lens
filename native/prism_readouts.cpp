// Small ABI bridge to the pinned Prism DLLs. No CUDA kernels are rebuilt.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
static thread_local std::string error;
struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    mtmd_context * vision = nullptr;
    std::string projector;
    int threads = 14, batch = 512;
    bool vision_gpu = false;
    bool capture_enabled = false;
    llama_pos position = 0;
    ~Engine() {
        if (vision) mtmd_free(vision);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    void capture(bool enabled) {
        if (enabled == capture_enabled) return;
        int32_t layer = llama_model_n_layer(model) - 1;
        llama_set_capture_layers(ctx, &layer, enabled ? 1 : 0);
        // Qwen3.5's h_nextn is assigned AFTER output_norm in qwen35.cpp.
        // Masked nextn extraction preserves last-token-only output. The generic
        // embeddings mode would request every prompt token and alter batching.
        llama_set_embeddings_nextn(ctx, enabled, true);
        capture_enabled = enabled;
    }
};

API const char * bonsai_error() { return error.c_str(); }
API void * bonsai_open(const char * directory, const char * model_path,
                       const char * projector, int context, int threads,
                       int batch, int ubatch, int gpu_layers,
                       const char * kv_type, bool vision_gpu) {
    try {
        error.clear();
        ggml_backend_load_all_from_path(directory);
        llama_backend_init();
        auto e = std::make_unique<Engine>();
        e->threads = threads;
        e->batch = batch;
        e->projector = projector;
        e->vision_gpu = vision_gpu;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        mp.load_mtp = false;
        e->model = llama_model_load_from_file(model_path, mp);
        if (!e->model) throw std::runtime_error("Prism model load failed");
        if (llama_model_n_embd(e->model) != 5120 ||
            llama_vocab_n_tokens(llama_model_get_vocab(e->model)) != 248320 ||
            llama_model_n_layer(e->model) != 64)
            throw std::runtime_error("Expected Ternary Bonsai 2 27B dimensions");
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = batch;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.no_perf = true;
        if (!std::strcmp(kv_type, "f16")) cp.type_k = cp.type_v = GGML_TYPE_F16;
        else if (!std::strcmp(kv_type, "q8_0")) cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        else if (!std::strcmp(kv_type, "q4_0")) cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
        else throw std::runtime_error("Native readouts support f16, q8_0 or q4_0 KV cache");
        e->ctx = llama_init_from_model(e->model, cp);
        if (!e->ctx) throw std::runtime_error("Prism context load failed");
        return e.release();
    } catch (const std::exception & ex) { error = ex.what(); return nullptr; }
}
API void bonsai_close(void * handle) { delete static_cast<Engine *>(handle); }
API void bonsai_reset(void * handle) {
    auto & e = *static_cast<Engine *>(handle);
    llama_memory_clear(llama_get_memory(e.ctx), true);
    e.position = 0;
}
API int bonsai_position(void * handle) { return static_cast<Engine *>(handle)->position; }
API int bonsai_decode(void * handle, const int32_t * tokens, int count, bool capture) {
    try {
        error.clear();
        auto & e = *static_cast<Engine *>(handle);
        if (count <= 0) throw std::runtime_error("No input tokens");
        if (e.position + count > int(llama_n_ctx(e.ctx)))
            throw std::runtime_error("Native context limit exceeded");
        e.capture(capture);
        auto batch = llama_batch_init(std::min(e.batch, count), 0, 1);
        int result = 0;
        for (int start = 0; start < count && !result; start += e.batch) {
            batch.n_tokens = std::min(e.batch, count - start);
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = tokens[start + i];
                batch.pos[i] = e.position + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == batch.n_tokens - 1);
            }
            result = llama_decode(e.ctx, batch);
            if (!result) e.position += batch.n_tokens;
        }
        llama_batch_free(batch);
        if (result) throw std::runtime_error("Prism decode failed: " + std::to_string(result));
        return 0;
    } catch (const std::exception & ex) { error = ex.what(); return -1; }
}
API float * bonsai_logits(void * handle) {
    return llama_get_logits_ith(static_cast<Engine *>(handle)->ctx, -1);
}
API float * bonsai_pre_norm(void * handle) {
    return llama_get_embeddings_capture_ith(static_cast<Engine *>(handle)->ctx, -1);
}
API float * bonsai_post_norm(void * handle) {
    return llama_get_embeddings_nextn_ith(static_cast<Engine *>(handle)->ctx, -1);
}
// Save only when switching between Lens and Chat sessions, never per token.
API size_t bonsai_state_size(void * handle) {
    return llama_state_seq_get_size(static_cast<Engine *>(handle)->ctx, 0);
}
API size_t bonsai_state_get(void * handle, uint8_t * out, size_t size) {
    return llama_state_seq_get_data(static_cast<Engine *>(handle)->ctx, out, size, 0);
}
API size_t bonsai_state_set(void * handle, const uint8_t * data, size_t size, int position) {
    auto & e = *static_cast<Engine *>(handle);
    auto result = llama_state_seq_set_data(e.ctx, data, size, 0);
    if (result) e.position = position;
    return result;
}
API int bonsai_image_prompt(void * handle, const char * prompt,
                           const char ** paths, int count, bool capture) {
    try {
        error.clear();
        auto & e = *static_cast<Engine *>(handle);
        if (e.projector.empty()) throw std::runtime_error("Vision projector missing");
        if (!e.vision) {
            auto vp = mtmd_context_params_default();
            vp.use_gpu = e.vision_gpu;
            vp.n_threads = e.threads;
            vp.warmup = false;
            e.vision = mtmd_init_from_file(e.projector.c_str(), e.model, vp);
            if (!e.vision) throw std::runtime_error("Vision load failed");
        }
        e.capture(capture);
        std::vector<mtmd_bitmap *> images;
        for (int i = 0; i < count; ++i) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(e.vision, paths[i], false);
            if (!wrapper.bitmap) {
                for (auto image : images) mtmd_bitmap_free(image);
                throw std::runtime_error("Cannot open input image");
            }
            images.push_back(wrapper.bitmap);
        }
        auto chunks = mtmd_input_chunks_init();
        mtmd_input_text text{prompt, std::char_traits<char>::length(prompt), false, true};
        int result = mtmd_tokenize(e.vision, chunks, &text,
            const_cast<const mtmd_bitmap **>(images.data()), images.size());
        if (!result && e.position + mtmd_helper_get_n_pos(chunks) > llama_n_ctx(e.ctx)) {
            result = -2;
        }
        if (!result) result = mtmd_helper_eval_chunks(e.vision, e.ctx, chunks,
            e.position, 0, e.batch, true, &e.position);
        mtmd_input_chunks_free(chunks);
        for (auto image : images) mtmd_bitmap_free(image);
        if (result) throw std::runtime_error("Native image prompt failed: " + std::to_string(result));
        return 0;
    } catch (const std::exception & ex) { error = ex.what(); return -1; }
}
