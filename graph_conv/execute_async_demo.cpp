// 计时版：证明 cudnnBackendExecute 只是"把 kernel 下发到 stream"，返回 ≠ 算完。
//   N=1 C=128 H=1024 W=1024, K=128 R=3 S=3, pad=1 stride=1
// 全程只经过 6 个 cudnnBackend* 入口（CreateDescriptor / SetAttribute /
// GetAttribute / Finalize / Execute / DestroyDescriptor），与 PyTorch frontend
// 最终落到的那 6 个符号完全一致。
#include <cudnn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr)                                                              \
  do {                                                                        \
    cudnnStatus_t st_ = (expr);                                               \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                        \
      std::printf("  [cuDNN 失败] %s\n      %s:%d -> %s\n", #expr, __FILE__,  \
                  __LINE__, cudnnGetErrorString(st_));                        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CU(expr)                                                              \
  do {                                                                        \
    cudaError_t er_ = (expr);                                                 \
    if (er_ != cudaSuccess) {                                                 \
      std::printf("  [CUDA 失败] %s: %s\n", #expr, cudaGetErrorString(er_));  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

struct Tally {
  int create = 0, destroy = 0, set = 0, get = 0, finalize = 0, execute = 0;
};
Tally g_tally;

constexpr int64_t N = 1, C = 128, H = 1024, W = 1024;
constexpr int64_t K = 128, R = 3, S = 3;
constexpr int64_t OH = H, OW = W;

int64_t g_xUid = 0, g_wUid = 1, g_yUid = 2;

cudnnBackendDescriptor_t makeTensor(int64_t uid, const int64_t dims[4],
                                    const int64_t strides[4], const char* what) {
  cudnnBackendDescriptor_t d = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &d));
  ++g_tally.create;
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE,
                              CUDNN_TYPE_DATA_TYPE, 1, &dt));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64,
                              4, dims));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4,
                              strides));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64,
                              1, &uid));
  int64_t align = 16;
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                              CUDNN_TYPE_INT64, 1, &align));
  g_tally.set += 5;
  CK(cudnnBackendFinalize(d));
  ++g_tally.finalize;
  std::printf("  张量 %-8s uid=%lld dims=[%lld,%lld,%lld,%lld]\n", what,
              (long long)uid, (long long)dims[0], (long long)dims[1],
              (long long)dims[2], (long long)dims[3]);
  return d;
}

void setAttr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
             cudnnBackendAttributeType_t type, int64_t count,
             const void* data) {
  CK(cudnnBackendSetAttribute(d, name, type, count, const_cast<void*>(data)));
  ++g_tally.set;
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

int main() {
  std::printf("== 1. 句柄与流（legacy 工具 API，不属于 graph API）\n");
  cudnnHandle_t handle = nullptr;
  CK(cudnnCreate(&handle));
  cudaStream_t stream = nullptr;
  CU(cudaStreamCreate(&stream));
  CK(cudnnSetStream(handle, stream));

  std::printf("\n== 2. 建图：三个张量描述符\n");
  const int64_t xDims[4] = {N, C, H, W};
  const int64_t xStr[4] = {C * H * W, H * W, W, 1};
  const int64_t wDims[4] = {K, C, R, S};
  const int64_t wStr[4] = {C * R * S, R * S, S, 1};
  const int64_t yDims[4] = {N, K, OH, OW};
  const int64_t yStr[4] = {K * OH * OW, OH * OW, OW, 1};
  cudnnBackendDescriptor_t xDesc = makeTensor(g_xUid, xDims, xStr, "x");
  cudnnBackendDescriptor_t wDesc = makeTensor(g_wUid, wDims, wStr, "w");
  cudnnBackendDescriptor_t yDesc = makeTensor(g_yUid, yDims, yStr, "y");

  std::printf("\n== 3. 建图：卷积描述符（不是 tensor descriptor）\n");
  cudnnBackendDescriptor_t convDesc = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  &convDesc));
  ++g_tally.create;
  cudnnDataType_t compType = CUDNN_DATA_FLOAT;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t pad[2] = {1, 1}, filtStride[2] = {1, 1}, dil[2] = {1, 1};
  const int64_t spatialDims = 2;
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1,
          &compType);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE,
          1, &mode);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1,
          &spatialDims);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
  setAttr(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2,
          filtStride);
  CK(cudnnBackendFinalize(convDesc));
  ++g_tally.finalize;

  std::printf("\n== 4. 建图：算子节点 + 操作图\n");
  cudnnBackendDescriptor_t opNode = nullptr;
  CK(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &opNode));
  ++g_tally.create;
  float alpha = 1.0f, beta = 0.0f;
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
          CUDNN_TYPE_FLOAT, 1, &alpha);
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT,
          1, &beta);
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc);
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc);
  setAttr(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc);
  CK(cudnnBackendFinalize(opNode));
  ++g_tally.finalize;

  cudnnBackendDescriptor_t opGraph = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  &opGraph));
  ++g_tally.create;
  setAttr(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1,
          &handle);
  setAttr(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
          1, &opNode);
  CK(cudnnBackendFinalize(opGraph));
  ++g_tally.finalize;

  std::printf("\n== 5. 引擎选择（heuristics）\n");
  cudnnBackendDescriptor_t heur = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
  ++g_tally.create;
  cudnnBackendHeurMode_t heurMode = CUDNN_HEUR_MODE_A;
  setAttr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph);
  setAttr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heurMode);
  CK(cudnnBackendFinalize(heur));
  ++g_tally.finalize;

  int64_t nCfg = 0;
  CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                              CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &nCfg, nullptr));
  ++g_tally.get;
  if (nCfg == 0) {
    std::printf("  HEUR_MODE_A 没有返回引擎，改用 FALLBACK\n");
    heurMode = CUDNN_HEUR_MODE_FALLBACK;
    setAttr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heurMode);
    CK(cudnnBackendFinalize(heur));
    ++g_tally.finalize;
    CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &nCfg,
                                nullptr));
    ++g_tally.get;
  }
  std::vector<cudnnBackendDescriptor_t> engCfgs(nCfg, nullptr);
  for (auto& c : engCfgs) {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &c));
    ++g_tally.create;
  }
  int64_t nGot = 0;
  CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                              CUDNN_TYPE_BACKEND_DESCRIPTOR, nCfg, &nGot,
                              engCfgs.data()));
  ++g_tally.get;
  std::printf("  heuristics 返回 %lld 个 engine config（实际取回 %lld 个，首个句柄=%p）\n",
              (long long)nCfg, (long long)nGot, (void*)engCfgs[0]);
  if (nGot < 1 || engCfgs[0] == nullptr) {
    std::printf("  engine config 列表为空，无法继续\n");
    return 1;
  }


  std::printf("\n== 6. 执行计划与 workspace\n");
  cudnnBackendDescriptor_t plan = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  &plan));
  ++g_tally.create;
  setAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  setAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfgs[0]);
  CK(cudnnBackendFinalize(plan));
  ++g_tally.finalize;
  int64_t wsSize = 0;
  CK(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                              CUDNN_TYPE_INT64, 1, nullptr, &wsSize));
  ++g_tally.get;
  std::printf("  workspace = %lld 字节\n", (long long)wsSize);

  std::printf("\n== 7. 数据上卡 + variant pack + 执行\n");
  std::vector<float> hx(N * C * H * W), hw(K * C * R * S), hy(N * K * OH * OW, 0.f);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = std::sin((float)i) * 0.5f;
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = std::cos((float)i) * 0.25f;

  void *dX = nullptr, *dW = nullptr, *dY = nullptr, *dWs = nullptr;
  CU(cudaMalloc(&dX, hx.size() * sizeof(float)));
  CU(cudaMalloc(&dW, hw.size() * sizeof(float)));
  CU(cudaMalloc(&dY, hy.size() * sizeof(float)));
  CU(cudaMemset(dY, 0, hy.size() * sizeof(float)));
  if (wsSize > 0) CU(cudaMalloc(&dWs, wsSize));
  CU(cudaMemcpy(dX, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dW, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudnnBackendDescriptor_t varPack = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                  &varPack));
  ++g_tally.create;
  const int64_t uids[3] = {g_xUid, g_wUid, g_yUid};
  void* const devPtrs[3] = {dX, dW, dY};
  setAttr(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
  setAttr(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3,
          devPtrs);
  setAttr(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
          &dWs);
  CK(cudnnBackendFinalize(varPack));
  ++g_tally.finalize;

  CK(cudnnBackendExecute(handle, plan, varPack));
  ++g_tally.execute;
  CU(cudaStreamSynchronize(stream));

  std::printf("\n== 8. 计时：cudnnBackendExecute 返回 vs 算完\n");
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto us = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };
  // 第一次是冷启动
  auto t0 = now();
  CK(cudnnBackendExecute(handle, plan, varPack)); ++g_tally.execute;
  auto t1 = now();
  CU(cudaStreamSynchronize(stream));
  auto t2 = now();
  std::printf("  第 1 次：execute 返回 %lld us，之后 synchronize 又等了 %lld us\n",
              (long long)us(t0, t1), (long long)us(t1, t2));
  // 再跑 3 次取稳定值
  long long hostSum = 0, devSum = 0;
  const int kReps = 3;
  for (int i = 0; i < kReps; ++i) {
    auto a = now();
    CK(cudnnBackendExecute(handle, plan, varPack)); ++g_tally.execute;
    auto b = now();
    CU(cudaStreamSynchronize(stream));
    auto c = now();
    hostSum += us(a, b);
    devSum += us(b, c);
  }
  std::printf("  后 %d 次：execute 返回平均 %lld us，GPU 上真正跑完还要 %lld us\n",
              kReps, hostSum / kReps, devSum / kReps);

  std::printf("\n== 9. 收尾\n");
  cudnnBackendDescriptor_t all[] = {varPack, plan, heur, opGraph, opNode,
                                    convDesc, yDesc, wDesc, xDesc};
  for (cudnnBackendDescriptor_t d : all) {
    if (d) {
      CK(cudnnBackendDestroyDescriptor(d));
      ++g_tally.destroy;
    }
  }
  for (cudnnBackendDescriptor_t c : engCfgs) {
    if (c) {
      CK(cudnnBackendDestroyDescriptor(c));
      ++g_tally.destroy;
    }
  }
  CU(cudaFree(dX));
  CU(cudaFree(dW));
  CU(cudaFree(dY));
  if (dWs) CU(cudaFree(dWs));
  CU(cudaStreamDestroy(stream));

  std::printf("\n== 本次程序实际用到的 graph API 入口统计 ==\n");
  std::printf("  cudnnBackendCreateDescriptor  %3d\n", g_tally.create);
  std::printf("  cudnnBackendSetAttribute      %3d\n", g_tally.set);
  std::printf("  cudnnBackendGetAttribute      %3d\n", g_tally.get);
  std::printf("  cudnnBackendFinalize          %3d\n", g_tally.finalize);
  std::printf("  cudnnBackendExecute           %3d\n", g_tally.execute);
  std::printf("  cudnnBackendDestroyDescriptor %3d\n", g_tally.destroy);
  return 0;
}
