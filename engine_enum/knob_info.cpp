// 枚举卷积图上每个 engine 暴露的旋钮：类型 / 最小值 / 最大值 / 步长。
// 用法: ./knob_info fwd [起始 engine 编号]
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
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

const char* knobName(int v) {
  switch (v) {
    case CUDNN_KNOB_TYPE_SPLIT_K: return "SPLIT_K";
    case CUDNN_KNOB_TYPE_SWIZZLE: return "SWIZZLE";
    case CUDNN_KNOB_TYPE_TILE_SIZE: return "TILE_SIZE";
    case CUDNN_KNOB_TYPE_USE_TEX: return "USE_TEX";
    case CUDNN_KNOB_TYPE_EDGE: return "EDGE";
    case CUDNN_KNOB_TYPE_KBLOCK: return "KBLOCK";
    case CUDNN_KNOB_TYPE_LDGA: return "LDGA";
    case CUDNN_KNOB_TYPE_LDGB: return "LDGB";
    case CUDNN_KNOB_TYPE_CHUNK_K: return "CHUNK_K";
    case CUDNN_KNOB_TYPE_SPLIT_H: return "SPLIT_H";
    case CUDNN_KNOB_TYPE_WINO_TILE: return "WINO_TILE";
    case CUDNN_KNOB_TYPE_MULTIPLY: return "MULTIPLY";
    case CUDNN_KNOB_TYPE_SPLIT_K_BUF: return "SPLIT_K_BUF";
    case CUDNN_KNOB_TYPE_TILEK: return "TILEK";
    case CUDNN_KNOB_TYPE_STAGES: return "STAGES";
    case CUDNN_KNOB_TYPE_REDUCTION_MODE: return "REDUCTION_MODE";
    case CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE: return "CTA_SPLIT_K_MODE";
    case CUDNN_KNOB_TYPE_SPLIT_K_SLC: return "SPLIT_K_SLC";
    case CUDNN_KNOB_TYPE_IDX_MODE: return "IDX_MODE";
    case CUDNN_KNOB_TYPE_SLICED: return "SLICED";
    case CUDNN_KNOB_TYPE_SPLIT_RS: return "SPLIT_RS";
    case CUDNN_KNOB_TYPE_SINGLEBUFFER: return "SINGLEBUFFER";
    case CUDNN_KNOB_TYPE_LDGC: return "LDGC";
    case CUDNN_KNOB_TYPE_SPECFILT: return "SPECFILT";
    case CUDNN_KNOB_TYPE_KERNEL_CFG: return "KERNEL_CFG";
    case CUDNN_KNOB_TYPE_WORKSPACE: return "WORKSPACE";
    case CUDNN_KNOB_TYPE_TILE_CGA: return "TILE_CGA";
    case CUDNN_KNOB_TYPE_TILE_CGA_M: return "TILE_CGA_M";
    case CUDNN_KNOB_TYPE_TILE_CGA_N: return "TILE_CGA_N";
    case CUDNN_KNOB_TYPE_BLOCK_SIZE: return "BLOCK_SIZE";
    case CUDNN_KNOB_TYPE_OCCUPANCY: return "OCCUPANCY";
    case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD: return "ARRAY_SIZE_PER_THREAD";
    case CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK: return "NUM_C_PER_BLOCK";
    case CUDNN_KNOB_TYPE_SPLIT_COLS: return "SPLIT_COLS";
    case CUDNN_KNOB_TYPE_TILE_ROWS: return "TILE_ROWS";
    case CUDNN_KNOB_TYPE_TILE_COLS: return "TILE_COLS";
    case CUDNN_KNOB_TYPE_LOAD_SIZE: return "LOAD_SIZE";
    case CUDNN_KNOB_TYPE_CTA_COUNT: return "CTA_COUNT";
    case CUDNN_KNOB_TYPE_STREAM_K: return "STREAM_K";
    case CUDNN_KNOB_TYPE_SPLIT_P_SLC: return "SPLIT_P_SLC";
    case CUDNN_KNOB_TYPE_TILE_M: return "TILE_M";
    case CUDNN_KNOB_TYPE_TILE_N: return "TILE_N";
    case CUDNN_KNOB_TYPE_WARP_SPEC_CFG: return "WARP_SPEC_CFG";
    case CUDNN_KNOB_TYPE_SWAP_AB: return "SWAP_AB";
    case CUDNN_KNOB_TYPE_INPUT_TMA_ENABLE: return "INPUT_TMA_ENABLE";
    case CUDNN_KNOB_TYPE_OUTPUT_TMA_ENABLE: return "OUTPUT_TMA_ENABLE";
    default: return "UNKNOWN_KNOB";
  }
}

bool getI64(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t a, int64_t* out) {
  int64_t n = 0, v = 0;
  if (cudnnBackendGetAttribute(d, a, CUDNN_TYPE_INT64, 1, &n, &v) != CUDNN_STATUS_SUCCESS || n < 1)
    return false;
  *out = v;
  return true;
}

bool getKnobType(cudnnBackendDescriptor_t d, int64_t* out) {
  int64_t n = 0, v = 0;
  if (cudnnBackendGetAttribute(d, CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &n, &v) !=
          CUDNN_STATUS_SUCCESS ||
      n < 1)
    return false;
  *out = v;
  return true;
}

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
  int start = argc > 2 ? std::atoi(argv[2]) : 0;
  int end = argc > 3 ? std::atoi(argv[3]) : -1;

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

  const std::string mode = argc > 1 ? argv[1] : "fwd";
  const bool bwd_data = mode == "bwd_data";
  const bool bwd_filter = mode == "bwd_filter";
  cudnnBackendDescriptor_t op = nullptr;
  float alpha = 1.f, beta = 0.f;
  auto setD = [](cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t a, cudnnBackendDescriptor_t v) {
    CK(cudnnBackendSetAttribute(d, a, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &v));
  };
  if (bwd_data) {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, &op));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, con);
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  } else if (bwd_filter) {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, &op));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, con);
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  } else {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, con);
  }
  CK(cudnnBackendFinalize(op));

  cudnnBackendDescriptor_t g = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &g));
  CK(cudnnBackendSetAttribute(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h));
  CK(cudnnBackendSetAttribute(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  CK(cudnnBackendFinalize(g));
  int64_t nEngCount = 0, total = 0;
  CK(cudnnBackendGetAttribute(g, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1,
                             &nEngCount, &total));
  std::printf("图 [N=%lld C=%lld %lldx%lld -> K=%lld]：engine 数 = %lld\n", (long long)N, (long long)C,
              (long long)H, (long long)W, (long long)K, (long long)total);

  if (end < 0 || end > total) end = int(total);
  for (int64_t i = start; i < end; ++i) {
    cudnnBackendDescriptor_t e = nullptr;
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &e));
    CK(cudnnBackendSetAttribute(e, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g));
    CK(cudnnBackendSetAttribute(e, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &i));
    cudnnStatus_t sf = cudnnBackendFinalize(e);
    std::printf("engine %-3lld finalize=%s", (long long)i, cudnnGetErrorString(sf));
    std::fflush(stdout);
    if (sf != CUDNN_STATUS_SUCCESS) { std::printf("\n"); cudnnBackendDestroyDescriptor(e); continue; }

    // 旋钮个数：先试 nullptr，再试非空探针（两种口径都见过）
    int64_t nk = -1;
    cudnnStatus_t sc = cudnnBackendGetAttribute(e, CUDNN_ATTR_ENGINE_KNOB_INFO, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0,
                                              &nk, nullptr);
    std::printf(" | 计数查询(空指针) st=%s nk=%lld", cudnnGetErrorString(sc), (long long)nk);
    std::fflush(stdout);
    std::printf(" | 旋钮数=%lld", (long long)nk);
    std::fflush(stdout);
    if (nk > 0) {
      std::vector<cudnnBackendDescriptor_t> knobs(size_t(nk), nullptr);
      // 关键：取回"描述符数组"前必须先自行创建每个 knob-info 描述符，
      // 否则 cuDNN 9.24 会在 libcudnn_graph 内部 get_internal() 段错误。
      for (int64_t k = 0; k < nk; ++k)
        CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR, &knobs[size_t(k)]));
      int64_t got = 0;
      cudnnStatus_t s = cudnnBackendGetAttribute(e, CUDNN_ATTR_ENGINE_KNOB_INFO, CUDNN_TYPE_BACKEND_DESCRIPTOR, nk, &got, knobs.data());
      std::printf(" | 取回 st=%s got=%lld\n", cudnnGetErrorString(s), (long long)got);
      std::fflush(stdout);
      for (int64_t k = 0; k < got && k < nk; ++k) {
        int64_t type = -1, mn = 0, mx = 0, stride = 0;
        bool okT = getKnobType(knobs[k], &type);
        bool okMn = getI64(knobs[k], CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, &mn);
        bool okMx = getI64(knobs[k], CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, &mx);
        bool okSt = getI64(knobs[k], CUDNN_ATTR_KNOB_INFO_STRIDE, &stride);
        std::printf("        %-22s 取值 [%lld..%lld] 步长 %lld   (%d%d%d%d)\n",
                    okT ? knobName(int(type)) : "?", (long long)mn, (long long)mx,
                    (long long)stride, int(okT), int(okMn), int(okMx), int(okSt));
        std::fflush(stdout);
      }
      for (int64_t k = 0; k < nk; ++k) cudnnBackendDestroyDescriptor(knobs[size_t(k)]);
    } else {
      std::printf("  （无旋钮 / 属性不可读）\n");
      std::fflush(stdout);
    }
    cudnnBackendDestroyDescriptor(e);
  }
  return 0;
}
