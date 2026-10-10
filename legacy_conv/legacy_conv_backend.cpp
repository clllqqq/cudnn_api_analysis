// 不用 graph API：只用 legacy 入口完成一次 Conv2d 前向。
//   N=1 C=3 H=8 W=8, K=4 R=3 S=3, pad=1 stride=1  ->  Y = 1x4x8x8
// 入口只有：cudnnCreate/SetStream + 3 类描述符 + 算法选择 + cudnnConvolutionForward。
// 用法：
//   ./legacy_conv_backend            # 启发式：cudnnGetConvolutionForwardAlgorithm_v7（取 1 个）
//   ./legacy_conv_backend --all      # 启发式：v7 直接要全部 8 个算法
//   ./legacy_conv_backend --find     # 实测：cudnnFindConvolutionForwardAlgorithmEx
#include <cudnn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(expr)                                                             \
  do {                                                                       \
    cudnnStatus_t st_ = (expr);                                              \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                       \
      std::printf("  [cuDNN 失败] %s\n      %s:%d -> %s\n", #expr, __FILE__, \
                  __LINE__, cudnnGetErrorString(st_));                       \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

#define CU(expr)                                                             \
  do {                                                                       \
    cudaError_t er_ = (expr);                                                \
    if (er_ != cudaSuccess) {                                                \
      std::printf("  [CUDA 失败] %s: %s\n", #expr, cudaGetErrorString(er_)); \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

struct Tally {
  int create = 0, destroy = 0, set = 0;
  int get_algo = 0, find_algo = 0, get_ws = 0, forward = 0;
};
Tally g_tally;

constexpr int64_t N = 1, C = 3, H = 8, W = 8;
constexpr int64_t K = 4, R = 3, S = 3;
constexpr int64_t OH = H, OW = W;
constexpr int64_t NUM_ALGOS = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;  // 8

const char* algoName(cudnnConvolutionFwdAlgo_t a) {
  switch (a) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      return "IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "WINOGRAD_NONFUSED";
    default: return "UNKNOWN";
  }
}

void conv2dReference(const std::vector<float>& x, const std::vector<float>& w,
                     std::vector<float>& y) {
  for (int64_t k = 0; k < K; ++k)
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        float acc = 0.f;
        for (int64_t c = 0; c < C; ++c)
          for (int64_t r = 0; r < R; ++r)
            for (int64_t s = 0; s < S; ++s) {
              int64_t ih = oh + r - 1, iw = ow + s - 1;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              acc += x[(c * H + ih) * W + iw] * w[((k * C + c) * R + r) * S + s];
            }
        y[(k * OH + oh) * OW + ow] = acc;
      }
}

}  // namespace

int main(int argc, char** argv) {
  bool use_find = false, use_all = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--find")) use_find = true;
    else if (!std::strcmp(argv[i], "--all")) use_all = true;
    else { std::printf("未知参数 %s\n", argv[i]); return 2; }
  }

  std::printf("== 1. 句柄 + 流（legacy 工具 API）\n");
  cudnnHandle_t handle = nullptr;
  CK(cudnnCreate(&handle)); ++g_tally.create;
  cudaStream_t stream = nullptr;
  CU(cudaStreamCreate(&stream));
  CK(cudnnSetStream(handle, stream)); ++g_tally.set;

  std::printf("\n== 2. 三类 legacy 描述符（没有 backend descriptor）\n");
  cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
  cudnnFilterDescriptor_t wDesc = nullptr;
  cudnnConvolutionDescriptor_t convDesc = nullptr;
  CK(cudnnCreateTensorDescriptor(&xDesc)); ++g_tally.create;
  CK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                (int)N, (int)C, (int)H, (int)W));
  ++g_tally.set;
  CK(cudnnCreateFilterDescriptor(&wDesc)); ++g_tally.create;
  CK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                (int)K, (int)C, (int)R, (int)S));
  ++g_tally.set;
  CK(cudnnCreateTensorDescriptor(&yDesc)); ++g_tally.create;
  CK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                (int)N, (int)K, (int)OH, (int)OW));
  ++g_tally.set;
  CK(cudnnCreateConvolutionDescriptor(&convDesc)); ++g_tally.create;
  CK(cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1,
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  ++g_tally.set;
  CK(cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH)); ++g_tally.set;
  std::printf("  x=1x3x8x8  w=4x3x3x3  y=1x4x8x8  pad=1 stride=1（CROSS_CORRELATION）\n");

  std::printf("\n== 3. 数据上卡\n");
  std::vector<float> hx(N * C * H * W), hw(K * C * R * S), hy(N * K * OH * OW, 0.f);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = std::sin((float)i) * 0.5f;
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = std::cos((float)i) * 0.25f;
  void *dX = nullptr, *dW = nullptr, *dY = nullptr, *dWs = nullptr;
  CU(cudaMalloc(&dX, hx.size() * sizeof(float)));
  CU(cudaMalloc(&dW, hw.size() * sizeof(float)));
  CU(cudaMalloc(&dY, hy.size() * sizeof(float)));
  CU(cudaMemset(dY, 0, hy.size() * sizeof(float)));
  CU(cudaMemcpy(dX, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dW, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("\n== 4. 选算法（legacy 的\"选引擎\"）\n");
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(NUM_ALGOS);
  int returned = 0;
  if (use_find) {
    // 与 PyTorch 的 getMaxWorkspaceSize 一样：逐个算法查 workspace，不支持的跳过
    static const cudnnConvolutionFwdAlgo_t all[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    size_t maxWs = 0;
    int nSupported = 0;
    for (cudnnConvolutionFwdAlgo_t a : all) {
      size_t sz = 0;
      cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
          handle, xDesc, wDesc, convDesc, yDesc, a, &sz);
      if (st != CUDNN_STATUS_SUCCESS) {
        std::printf("    %-18s 不支持（%s），跳过\n", algoName(a),
                    cudnnGetErrorString(st));
        continue;
      }
      ++g_tally.get_ws;
      ++nSupported;
      maxWs = std::max(maxWs, sz);
    }
    if (nSupported == 0) { std::printf("  没有任何 legacy 算法可用\n"); return 1; }
    if (maxWs) CU(cudaMalloc(&dWs, maxWs));
    CK(cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, dX, wDesc, dW,
                                              convDesc, yDesc, dY, nSupported,
                                              &returned, perf.data(), dWs,
                                              maxWs));
    ++g_tally.find_algo;
    std::printf("  cudnnFindConvolutionForwardAlgorithmEx 实测 %d 个算法（workspace=%zu B）：\n",
                returned, maxWs);
    // 和 PyTorch 的 getValidAlgorithms 一样，只保留 status==SUCCESS 的结果
    std::vector<cudnnConvolutionFwdAlgoPerf_t> ok;
    for (int i = 0; i < returned; ++i)
      if (perf[i].status == CUDNN_STATUS_SUCCESS) ok.push_back(perf[i]);
    for (size_t i = 0; i < ok.size(); ++i) perf[i] = ok[i];
    returned = (int)ok.size();
    for (int i = 0; i < returned; ++i)
      std::printf("    #%d %-18s time=%8.4f ms  workspace=%zu\n", i + 1,
                  algoName(perf[i].algo), perf[i].time, perf[i].memory);
    if (returned < 1) { std::printf("  实测后没有可用算法\n"); return 1; }
  } else {
    const int want = use_all ? (int)NUM_ALGOS : 1;
    CK(cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc,
                                              yDesc, want, &returned,
                                              perf.data()));
    ++g_tally.get_algo;
    std::printf("  cudnnGetConvolutionForwardAlgorithm_v7 请求 %d 个，返回 %d 个：\n",
                want, returned);
    for (int i = 0; i < returned; ++i)
      std::printf("    #%d %-18s 预估 time=%8.4f ms  workspace=%zu\n", i + 1,
                  algoName(perf[i].algo), perf[i].time, perf[i].memory);
  }

  std::printf("\n== 5. workspace + cudnnConvolutionForward\n");
  // 依次尝试候选，取第一个"能拿到 workspace"的（PyTorch AlgoIterator 的做法）
  size_t wsSize = 0;
  int chosen = -1;
  for (int i = 0; i < returned; ++i) {
    if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
    size_t sz = 0;
    cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, perf[i].algo, &sz);
    if (st != CUDNN_STATUS_SUCCESS) continue;
    ++g_tally.get_ws;
    wsSize = sz;
    chosen = i;
    break;
  }
  if (chosen < 0) { std::printf("  候选算法都拿不到 workspace\n"); return 1; }
  const cudnnConvolutionFwdAlgo_t algo = perf[chosen].algo;
  if (wsSize && !dWs) CU(cudaMalloc(&dWs, wsSize));
  std::printf("  选中 %s，workspace=%zu 字节\n", algoName(algo), wsSize);

  const float alpha = 1.f, beta = 0.f;
  CU(cudaMemset(dY, 0, hy.size() * sizeof(float)));
  CK(cudnnConvolutionForward(handle, &alpha, xDesc, dX, wDesc, dW, convDesc,
                             algo, dWs, wsSize, &beta, yDesc, dY));
  ++g_tally.forward;
  CU(cudaStreamSynchronize(stream));

  std::printf("\n== 6. 结果校验（与 CPU 直接卷积对比）\n");
  CU(cudaMemcpy(hy.data(), dY, hy.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
  std::vector<float> ref(hy.size(), 0.f);
  conv2dReference(hx, hw, ref);
  float maxErr = 0.f;
  for (size_t i = 0; i < hy.size(); ++i)
    maxErr = std::max(maxErr, std::fabs(hy[i] - ref[i]));
  std::printf("  y[0][0][0][0]=%.6f  参考值=%.6f\n", hy[0], ref[0]);
  std::printf("  y[0][3][7][7]=%.6f  参考值=%.6f\n", hy[hy.size() - 1],
              ref[ref.size() - 1]);
  std::printf("  最大绝对误差 = %.3e  ->  %s\n", maxErr,
              maxErr < 1e-4f ? "通过" : "不通过");

  std::printf("\n== 7. 收尾\n");
  CU(cudaFree(dX)); CU(cudaFree(dW)); CU(cudaFree(dY));
  if (dWs) CU(cudaFree(dWs));
  CK(cudnnDestroyConvolutionDescriptor(convDesc)); ++g_tally.destroy;
  CK(cudnnDestroyFilterDescriptor(wDesc)); ++g_tally.destroy;
  CK(cudnnDestroyTensorDescriptor(yDesc)); ++g_tally.destroy;
  CK(cudnnDestroyTensorDescriptor(xDesc)); ++g_tally.destroy;
  CU(cudaStreamDestroy(stream));
  // PyTorch 在进程内不会调用它（destroyCuDNNHandle 是空函数，见文档 §10.2）
  CK(cudnnDestroy(handle)); ++g_tally.destroy;

  std::printf("\n== 本次程序实际用到的 legacy 入口统计 ==\n");
  std::printf("  cudnnGetConvolutionForwardAlgorithm_v7   %3d\n", g_tally.get_algo);
  std::printf("  cudnnFindConvolutionForwardAlgorithmEx   %3d\n", g_tally.find_algo);
  std::printf("  cudnnGetConvolutionForwardWorkspaceSize  %3d\n", g_tally.get_ws);
  std::printf("  cudnnConvolutionForward                  %3d\n", g_tally.forward);
  std::printf("  描述符 Create/Destroy/Set                %3d / %3d / %3d\n",
              g_tally.create, g_tally.destroy, g_tally.set);
  std::printf("  cudnnBackend* 入口                       0（全程没有 graph API）\n");
  return 0;
}
