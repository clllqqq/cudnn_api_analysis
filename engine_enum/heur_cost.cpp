// 测 heuristic 各模式的 CPU 代价：从建描述符到拿回候选全流程分段计时。
// 用法: ./heur_cost <mode 0..3> [reps]        （形状用 RUNNER_N/C/HW/K 改）
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#define CK(expr)                                                              \
  do {                                                                        \
    cudnnStatus_t st_ = (expr);                                               \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                        \
      std::printf("  [cuDNN 失败] %s -> %s\n", #expr, cudnnGetErrorString(st_));\
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {
int64_t N = 4, C = 3, H = 64, W = 64, K = 16;
constexpr int64_t R = 3, S = 3;

cudnnBackendDescriptor_t tensor(int64_t uid, const int64_t dims[4], const int64_t strides[4]) {
  cudnnBackendDescriptor_t d = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &d));
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  int64_t al = 16;
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, (void*)dims));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, (void*)strides));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &al));
  CK(cudnnBackendFinalize(d));
  return d;
}
}  // namespace


int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (const char* e = std::getenv("RUNNER_N")) N = std::atol(e);
  if (const char* e = std::getenv("RUNNER_C")) C = std::atol(e);
  if (const char* e = std::getenv("RUNNER_HW")) H = W = std::atol(e);
  if (const char* e = std::getenv("RUNNER_K")) K = std::atol(e);
  const int64_t OH = H, OW = W;
  const int mode = argc > 1 ? std::atoi(argv[1]) : 3;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 3;

  cudnnHandle_t h = nullptr;
  CK(cudnnCreate(&h));
  cudaStream_t st = nullptr;
  cudaStreamCreate(&st);
  CK(cudnnSetStream(h, st));

  const int64_t xd[4] = {N, C, H, W}, xs[4] = {C * H * W, H * W, W, 1};
  const int64_t wd[4] = {K, C, R, S}, ws[4] = {C * R * S, R * S, S, 1};
  const int64_t yd[4] = {N, K, OH, OW}, ys[4] = {K * OH * OW, OH * OW, OW, 1};
  cudnnBackendDescriptor_t x = tensor(0, xd, xs), w = tensor(1, wd, ws), y = tensor(2, yd, ys);

  cudnnBackendDescriptor_t con = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &con));
  cudnnConvolutionMode_t cm = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  int64_t sd = 2, p[2] = {1, 1};
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &cm));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &sd));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, p));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, p));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, p));
  CK(cudnnBackendSetAttribute(con, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, p));
  CK(cudnnBackendFinalize(con));

  cudnnBackendDescriptor_t op = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
  float alpha = 1.f, beta = 0.f;
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x));
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w));
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y));
  CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &con));
  CK(cudnnBackendFinalize(op));

  cudnnBackendDescriptor_t g = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &g));
  CK(cudnnBackendSetAttribute(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h));
  CK(cudnnBackendSetAttribute(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  CK(cudnnBackendFinalize(g));

  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // 可选：先用另一个 mode 做一次"热身"的 heuristic（用于区分 dlopen/首次初始化 与 真实计算）
  if (const char* e = std::getenv("HEUR_PRE")) {
    cudnnBackendDescriptor_t warm = nullptr;
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &warm));
    CK(cudnnBackendSetAttribute(warm, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g));
    cudnnBackendHeurMode_t wm = cudnnBackendHeurMode_t(std::atoi(e));
    CK(cudnnBackendSetAttribute(warm, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &wm));
    CK(cudnnBackendFinalize(warm));
    int64_t nw = 0;
    CK(cudnnBackendGetAttribute(warm, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &nw, nullptr));
    std::vector<cudnnBackendDescriptor_t> wc(size_t(nw), nullptr);
    for (auto& c : wc) CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &c));
    int64_t gw = 0;
    CK(cudnnBackendGetAttribute(warm, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, nw, &gw, wc.data()));
    for (auto c : wc) cudnnBackendDestroyDescriptor(c);
    cudnnBackendDestroyDescriptor(warm);
    std::printf("[热身 mode=%s -> %lld 个候选] ", e, (long long)gw);
  }
  double f_first = -1, f_min = 1e9, r_first = -1, r_min = 1e9, c_first = -1, c_min = 1e9;
  int64_t nCfg = 0, got = 0;
  for (int r = 0; r < reps; ++r) {
    cudnnBackendDescriptor_t heur = nullptr;
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
    CK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g));
    cudnnBackendHeurMode_t hmode = cudnnBackendHeurMode_t(mode);
    CK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hmode));
    auto t1 = clk::now();
    CK(cudnnBackendFinalize(heur));
    auto t2 = clk::now();
    CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &nCfg, nullptr));
    auto t3 = clk::now();
    std::vector<cudnnBackendDescriptor_t> cfgs(size_t(nCfg), nullptr);
    for (auto& c : cfgs) CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &c));
    CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, nCfg, &got, cfgs.data()));
    auto t4 = clk::now();
    for (auto c : cfgs) cudnnBackendDestroyDescriptor(c);
    cudnnBackendDestroyDescriptor(heur);
    const double vf = ms(t1, t2), vc = ms(t2, t3), vr = ms(t3, t4);
    if (r == 0) { f_first = vf; c_first = vc; r_first = vr; }
    f_min = std::min(f_min, vf);
    c_min = std::min(c_min, vc);
    r_min = std::min(r_min, vr);
  }
  std::printf("mode=%d shape=N%lldxC%lldx%lldx%lld->K%lld reps=%d cands=%lld | finalize first=%.3f/min=%.3fms | count first=%.4f/min=%.4fms | retrieve first=%.3f/min=%.3fms\n",
              mode, (long long)N, (long long)C, (long long)H, (long long)W, (long long)K, reps,
              (long long)got, f_first, f_min, c_first, c_min, r_first, r_min);
  return 0;
}
