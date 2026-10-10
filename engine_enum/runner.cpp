// 把某个 heuristic 候选 engine 真正跑起来（供 CUPTI/torch.profiler 抓 kernel）。
// 编译成动态库，Python 侧用 ctypes 调用；CUDA 活动由 CUPTI 进程级捕获。
//   int run_cand(int cand, int iters, char* json, int cap, double* ms, double* err)
#include <cudnn.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CK2(e, rc)                                                            \
  do {                                                                        \
    cudnnStatus_t s_ = (e);                                                   \
    if (s_ != CUDNN_STATUS_SUCCESS) {                                         \
      std::printf("[runner] %s -> %s\n", #e, cudnnGetErrorString(s_));        \
      return (rc);                                                            \
    }                                                                         \
  } while (0)

namespace {

int64_t N = 4;
constexpr int64_t C = 3, H = 64, W = 64;
constexpr int64_t K = 16, R = 3, S = 3;
constexpr int64_t OH = 64, OW = 64;

struct Ctx {
  cudnnHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  cudnnBackendDescriptor_t graph = nullptr, heur = nullptr, conv = nullptr;
  cudnnBackendDescriptor_t x = nullptr, w = nullptr, y = nullptr, op = nullptr;
  std::vector<cudnnBackendDescriptor_t> cfgs;
  bool ready = false;
};

cudnnBackendDescriptor_t makeTensor(int64_t uid, const int64_t dims[4],
                                    const int64_t strides[4]) {
  cudnnBackendDescriptor_t d = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &d);
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  int64_t align = 16;
  cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
  cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, (void*)dims);
  cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, (void*)strides);
  cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  cudnnBackendFinalize(d);
  return d;
}

int build(Ctx& c) {
  if (c.ready) return 0;
  if (const char* e = std::getenv("RUNNER_N")) N = std::atol(e);
  CK2(cudnnCreate(&c.handle), -1);
  cudaStreamCreate(&c.stream);
  CK2(cudnnSetStream(c.handle, c.stream), -2);

  const int64_t xDims[4] = {N, C, H, W}, xStr[4] = {C * H * W, H * W, W, 1};
  const int64_t wDims[4] = {K, C, R, S}, wStr[4] = {C * R * S, R * S, S, 1};
  const int64_t yDims[4] = {N, K, OH, OW}, yStr[4] = {K * OH * OW, OH * OW, OW, 1};
  c.x = makeTensor(0, xDims, xStr);
  c.w = makeTensor(1, wDims, wStr);
  c.y = makeTensor(2, yDims, yStr);

  CK2(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &c.conv), -3);
  cudnnConvolutionMode_t cm = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  int64_t sd = 2, p[2] = {1, 1};
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &cm);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &sd);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, p);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, p);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, p);
  cudnnBackendSetAttribute(c.conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, p);
  CK2(cudnnBackendFinalize(c.conv), -4);

  CK2(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &c.op), -5);
  float alpha = 1.f, beta = 0.f;
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.x);
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.w);
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.y);
  cudnnBackendSetAttribute(c.op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.conv);
  CK2(cudnnBackendFinalize(c.op), -6);

  CK2(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &c.graph), -7);
  cudnnBackendSetAttribute(c.graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &c.handle);
  cudnnBackendSetAttribute(c.graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.op);
  CK2(cudnnBackendFinalize(c.graph), -8);

  CK2(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &c.heur), -9);
  cudnnBackendSetAttribute(c.heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c.graph);
  cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_A;
  cudnnBackendSetAttribute(c.heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
  CK2(cudnnBackendFinalize(c.heur), -10);
  int64_t ncfg = 0;
  CK2(cudnnBackendGetAttribute(c.heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &ncfg, nullptr), -11);
  c.cfgs.assign(size_t(ncfg), nullptr);
  for (auto& cc : c.cfgs) cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cc);
  int64_t got = 0;
  CK2(cudnnBackendGetAttribute(c.heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ncfg, &got, c.cfgs.data()), -12);
  c.cfgs.resize(size_t(got));
  c.ready = true;
  return 0;
}

void convRef(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y) {
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k)
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
          float acc = 0.f;
          for (int64_t c = 0; c < C; ++c)
            for (int64_t r = 0; r < R; ++r)
              for (int64_t s = 0; s < S; ++s) {
                int64_t ih = oh + r - 1, iw = ow + s - 1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                acc += x[((n * C + c) * H + ih) * W + iw] * w[((k * C + c) * R + r) * S + s];
              }
          y[((n * K + k) * OH + oh) * OW + ow] = acc;
        }
}

Ctx g_ctx;

int exec_plan(cudnnBackendDescriptor_t cfg, int iters, char* json_out, int json_cap,
              double* ms_out, double* err_out) {
  cudnnBackendDescriptor_t plan = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &g_ctx.handle);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
  cudnnStatus_t sp = cudnnBackendFinalize(plan);
  if (sp != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] plan finalize -> %s\n", cudnnGetErrorString(sp));
    return -31;
  }
  if (json_out && json_cap > 0) {
    int64_t len = 0;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, &len, nullptr);
    if (len > json_cap) len = json_cap;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, len, &len, json_out);
    json_out[len < json_cap ? len : json_cap - 1] = 0;
  }
  int64_t wsSize = 0, wsCount = 0;
  cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &wsCount, &wsSize);

  std::vector<float> hx(N * C * H * W), hw(K * C * R * S), hy(N * K * OH * OW, 0.f);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = std::sin(float(i) * 0.01f);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = std::cos(float(i) * 0.02f);
  void *dX = nullptr, *dW = nullptr, *dY = nullptr, *dWs = nullptr;
  cudaMalloc(&dX, hx.size() * 4);
  cudaMalloc(&dW, hw.size() * 4);
  cudaMalloc(&dY, hy.size() * 4);
  if (wsSize > 0) cudaMalloc(&dWs, wsSize);
  cudaMemcpy(dX, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(dY, 0, hy.size() * 4);
  cudnnBackendDescriptor_t vp = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp);
  const int64_t uids[3] = {0, 1, 2};
  void* const ptrs[3] = {dX, dW, dY};
  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, (void*)uids);
  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, (void*)ptrs);
  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &dWs);
  cudnnBackendFinalize(vp);
  cudnnStatus_t st = cudnnBackendExecute(g_ctx.handle, plan, vp);
  if (st != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] execute -> %s\n", cudnnGetErrorString(st));
    return -32;
  }
  cudaStreamSynchronize(g_ctx.stream);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, g_ctx.stream);
  for (int i = 0; i < iters; ++i) cudnnBackendExecute(g_ctx.handle, plan, vp);
  cudaEventRecord(e1, g_ctx.stream);
  cudaEventSynchronize(e1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  *ms_out = double(ms) / iters;
  cudaMemcpy(hy.data(), dY, hy.size() * 4, cudaMemcpyDeviceToHost);
  std::vector<float> ref(hy.size(), 0.f);
  convRef(hx, hw, ref);
  double maxErr = 0.0;
  for (size_t i = 0; i < hy.size(); ++i)
    maxErr = std::max(maxErr, double(std::fabs(double(hy[i]) - double(ref[i]))));
  *err_out = maxErr;
  cudaFree(dX); cudaFree(dW); cudaFree(dY);
  if (dWs) cudaFree(dWs);
  cudnnBackendDestroyDescriptor(vp);
  cudnnBackendDestroyDescriptor(plan);
  return 0;
}

}  // namespace

// cand: heuristic 候选序号；iters: 重复执行次数（用于计时/抓 kernel）
extern "C" int run_cand(int cand, int iters, char* json_out, int json_cap,
                        double* ms_out, double* err_out) {
  int rc = build(g_ctx);
  if (rc) return rc;
  if (cand < 0 || size_t(cand) >= g_ctx.cfgs.size()) return -20;
  return exec_plan(g_ctx.cfgs[cand], iters, json_out, json_cap, ms_out, err_out);
}

// 直接按"全局 engine 编号"建 engine config 并执行（用于对比预编译 / runtime 编译引擎）。
// plan_ms = 从建 engine config 到 plan finalize 完成的时间（runtime 引擎在这里现场编译）。
extern "C" int run_engine_global(int eng_id, int iters, char* json_out, int json_cap,
                                 double* plan_ms_out, double* ms_out, double* err_out) {
  int rc = build(g_ctx);
  if (rc) return rc;
  auto t0 = std::chrono::steady_clock::now();
  cudnnBackendDescriptor_t eng = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &eng);
  cudnnBackendSetAttribute(eng, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g_ctx.graph);
  int64_t idx = eng_id;
  cudnnBackendSetAttribute(eng, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &idx);
  cudnnStatus_t se = cudnnBackendFinalize(eng);
  if (se != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] engine %d finalize -> %s\n", eng_id, cudnnGetErrorString(se));
    return -40;
  }
  cudnnBackendDescriptor_t cfg = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg);
  cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &eng);
  cudnnStatus_t sc = cudnnBackendFinalize(cfg);
  if (sc != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] engine %d config finalize -> %s\n", eng_id, cudnnGetErrorString(sc));
    return -41;
  }
  // 先把 plan 建出来（这里会触发 runtime 引擎的 NVRTC 编译）
  cudnnBackendDescriptor_t probe = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &probe);
  cudnnBackendSetAttribute(probe, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &g_ctx.handle);
  cudnnBackendSetAttribute(probe, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
  cudnnBackendFinalize(probe);
  auto t1 = std::chrono::steady_clock::now();
  *plan_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (json_out && json_cap > 0) {
    int64_t len = 0;
    cudnnBackendGetAttribute(probe, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, &len, nullptr);
    if (len > json_cap) len = json_cap;
    cudnnBackendGetAttribute(probe, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, len, &len, json_out);
    json_out[len < json_cap ? len : json_cap - 1] = 0;
  }
  cudnnBackendDestroyDescriptor(probe);
  return exec_plan(cfg, iters, nullptr, 0, ms_out, err_out);
}

extern "C" int num_cands() {
  if (build(g_ctx)) return -1;
  return int(g_ctx.cfgs.size());
}

// 按"全局 engine 编号 + 手工旋钮"建 engine config 并执行（旋钮扫描用）。
// knob_types: cudnnBackendKnobType_t 数组；knob_vals: 对应取值；n_knobs: 个数。
extern "C" int run_engine_global_knobs(int eng_id, const int* knob_types, const long long* knob_vals,
                                       int n_knobs, int iters, char* json_out, int json_cap,
                                       double* plan_ms_out, double* ms_out, double* err_out) {
  int rc = build(g_ctx);
  if (rc) return rc;
  auto t0 = std::chrono::steady_clock::now();
  cudnnBackendDescriptor_t eng = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &eng);
  cudnnBackendSetAttribute(eng, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g_ctx.graph);
  int64_t idx = eng_id;
  cudnnBackendSetAttribute(eng, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &idx);
  cudnnStatus_t se = cudnnBackendFinalize(eng);
  if (se != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] engine %d finalize -> %s\n", eng_id, cudnnGetErrorString(se));
    return -40;
  }
  std::vector<cudnnBackendDescriptor_t> choices(size_t(n_knobs), nullptr);
  for (int i = 0; i < n_knobs; ++i) {
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, &choices[size_t(i)]);
    cudnnBackendKnobType_t kt = (cudnnBackendKnobType_t)knob_types[i];
    int64_t kv = knob_vals[i];
    cudnnBackendSetAttribute(choices[size_t(i)], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &kt);
    cudnnBackendSetAttribute(choices[size_t(i)], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &kv);
    cudnnStatus_t sk = cudnnBackendFinalize(choices[size_t(i)]);
    if (sk != CUDNN_STATUS_SUCCESS) {
      std::printf("[runner] engine %d knob %d=%lld -> %s\n", eng_id, knob_types[i], (long long)kv,
                  cudnnGetErrorString(sk));
      return -42;
    }
  }
  cudnnBackendDescriptor_t cfg = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg);
  cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &eng);
  if (n_knobs > 0)
    cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                             n_knobs, choices.data());
  cudnnStatus_t sc = cudnnBackendFinalize(cfg);
  if (sc != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] engine %d config finalize -> %s\n", eng_id, cudnnGetErrorString(sc));
    return -41;
  }
  cudnnBackendDescriptor_t plan = nullptr;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &g_ctx.handle);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
  cudnnStatus_t sp = cudnnBackendFinalize(plan);
  auto t1 = std::chrono::steady_clock::now();
  *plan_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (sp != CUDNN_STATUS_SUCCESS) {
    std::printf("[runner] engine %d plan finalize -> %s\n", eng_id, cudnnGetErrorString(sp));
    return -43;
  }
  if (json_out && json_cap > 0) {
    int64_t len = 0;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, &len, nullptr);
    if (len > json_cap) len = json_cap;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, len, &len, json_out);
    json_out[len < json_cap ? len : json_cap - 1] = 0;
  }
  cudnnBackendDestroyDescriptor(plan);
  return exec_plan(cfg, iters, nullptr, 0, ms_out, err_out);
}
