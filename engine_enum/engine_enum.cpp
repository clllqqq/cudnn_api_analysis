// 枚举 cuDNN 为某个卷积图提供的全部 engine，并打印每个 engine 的属性。
//   fwd       : x(4,3,64,64) * w(16,3,3,3) -> y(4,16,64,64)
//   bwd_data  : dy(4,16,64,64) * w(16,3,3,3) -> dx(4,3,64,64)
//   bwd_filter: dy(4,16,64,64) * x(4,3,64,64) -> dw(16,3,3,3)
// 形状取自 /home/liquan/cudnn_per_op.py 的 conv case，便于和 PyTorch 实测对齐。
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
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

namespace {

int64_t N = 4, C = 3, H = 64, W = 64;
int64_t K = 16;
constexpr int64_t R = 3, S = 3;

const char* numNoteName(int v) {
  switch (v) {
    case CUDNN_NUMERICAL_NOTE_TENSOR_CORE: return "TENSOR_CORE";
    case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS: return "DOWN_CONVERT_INPUTS";
    case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
      return "REDUCED_PRECISION_REDUCTION";
    case CUDNN_NUMERICAL_NOTE_FFT: return "FFT";
    case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC: return "NONDETERMINISTIC";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD: return "WINOGRAD";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4: return "WINOGRAD_TILE_4x4";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6: return "WINOGRAD_TILE_6x6";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13: return "WINOGRAD_TILE_13x13";
    case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP: return "STRICT_NAN_PROP";
    default: return "UNKNOWN_NUM";
  }
}

const char* behNoteName(int v) {
  switch (v) {
    case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION: return "RUNTIME_COMPILATION";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER: return "REQ_FILTER_INT8x32_REORDER";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER: return "REQ_BIAS_INT8x32_REORDER";
    case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API: return "SUPPORTS_CUDA_GRAPH_NATIVE_API";
    case CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY: return "CUBLASLT_DEPENDENCY";
    default: return "UNKNOWN_BEH";
  }
}

const char* layoutName(int v) {
  switch (v) {
    case CUDNN_LAYOUT_TYPE_PREFERRED_NCHW: return "NCHW";
    case CUDNN_LAYOUT_TYPE_PREFERRED_NHWC: return "NHWC";
    case CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK: return "PAD4CK";
    case CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK: return "PAD8CK";
    default: return "?";
  }
}

const char* knobName(int v) {
  switch (v) {
    case CUDNN_KNOB_TYPE_SWIZZLE: return "SWIZZLE";
    case CUDNN_KNOB_TYPE_TILE_SIZE: return "TILE_SIZE";
    case CUDNN_KNOB_TYPE_EDGE: return "EDGE";
    case CUDNN_KNOB_TYPE_MULTIPLY: return "MULTIPLY";
    case CUDNN_KNOB_TYPE_SPLIT_K_BUF: return "SPLIT_K_BUF";
    case CUDNN_KNOB_TYPE_TILEK: return "TILEK";
    case CUDNN_KNOB_TYPE_STAGES: return "STAGES";
    case CUDNN_KNOB_TYPE_REDUCTION_MODE: return "REDUCTION_MODE";
    case CUDNN_KNOB_TYPE_SPLIT_K_SLC: return "SPLIT_K_SLC";
    case CUDNN_KNOB_TYPE_IDX_MODE: return "IDX_MODE";
    case CUDNN_KNOB_TYPE_SPECFILT: return "SPECFILT";
    case CUDNN_KNOB_TYPE_KERNEL_CFG: return "KERNEL_CFG";
    case CUDNN_KNOB_TYPE_WORKSPACE: return "WORKSPACE";
    case CUDNN_KNOB_TYPE_TILE_CGA_M: return "TILE_CGA_M";
    case CUDNN_KNOB_TYPE_TILE_CGA_N: return "TILE_CGA_N";
    case CUDNN_KNOB_TYPE_BLOCK_SIZE: return "BLOCK_SIZE";
    case CUDNN_KNOB_TYPE_OCCUPANCY: return "OCCUPANCY";
    case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD: return "ARRAY_SIZE_PER_THREAD";
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
    default: return "?";
  }
}

struct AttrBuf {
  std::vector<unsigned char> bytes;
  int64_t count = 0;
  bool ok = false;
  template <typename T> const T* as() const { return reinterpret_cast<const T*>(bytes.data()); }
};

AttrBuf getAttr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                cudnnBackendAttributeType_t type, size_t elemSize) {
  AttrBuf out;
  int64_t n = 0;
  // count 查询两种口径：多数属性要 nullptr；个别属性（如 ENGINEHEUR_RESULTS）
  // 反而要非空 data 指针。两种都试一次，避免把"取不到个数"误当"没有元素"。
  cudnnStatus_t st = cudnnBackendGetAttribute(d, name, type, 0, &n, nullptr);
  if (st != CUDNN_STATUS_SUCCESS || n <= 0) {
    std::vector<unsigned char> probe(elemSize < 256 ? 256 : elemSize);
    int64_t n2 = 0;
    cudnnStatus_t st2 = cudnnBackendGetAttribute(d, name, type, 0, &n2, probe.data());
    if (st2 == CUDNN_STATUS_SUCCESS && n2 > 0) { st = st2; n = n2; }
  }
  if (st != CUDNN_STATUS_SUCCESS || n <= 0) return out;
  out.bytes.resize(size_t(n) * elemSize);
  st = cudnnBackendGetAttribute(d, name, type, n, &n, out.bytes.data());
  if (st != CUDNN_STATUS_SUCCESS) return out;
  out.count = n;
  out.ok = true;
  return out;
}

std::string noteList(const AttrBuf& a, const char* (*name)(int)) {
  std::string s;
  if (!a.ok) return "-";
  for (int64_t i = 0; i < a.count; ++i) {
    if (i) s += ",";
    s += name(int(a.as<int32_t>()[i]));
  }
  return s;
}

cudnnBackendDescriptor_t makeTensor(int64_t uid, const int64_t dims[4],
                                    const int64_t strides[4], cudnnDataType_t dt) {
  cudnnBackendDescriptor_t d = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &d));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides));
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
  int64_t align = 16;
  CK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align));
  CK(cudnnBackendFinalize(d));
  return d;
}

void setD(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t n, cudnnBackendDescriptor_t v) {
  CK(cudnnBackendSetAttribute(d, n, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &v));
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (const char* e = std::getenv("RUNNER_N")) N = std::atol(e);
  if (const char* e = std::getenv("RUNNER_C")) C = std::atol(e);
  if (const char* e = std::getenv("RUNNER_HW")) H = W = std::atol(e);
  if (const char* e = std::getenv("RUNNER_K")) K = std::atol(e);
  const int64_t OH = H, OW = W;
  const std::string mode = argc > 1 ? argv[1] : "fwd";

  cudnnHandle_t handle = nullptr;
  CK(cudnnCreate(&handle));
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  CK(cudnnSetStream(handle, stream));
  int dev = 0, ccMajor = 0, ccMinor = 0, smCount = 0, drv = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
  cudaDriverGetVersion(&drv);
  std::printf("设备: dev%d  sm_%d%d  SM 数=%d  驱动 %d.%d\n", dev, ccMajor, ccMinor,
              smCount, drv / 1000, (drv % 1000) / 10);

  const cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  const int64_t xDims[4] = {N, C, H, W}, xStr[4] = {C * H * W, H * W, W, 1};
  const int64_t wDims[4] = {K, C, R, S}, wStr[4] = {C * R * S, R * S, S, 1};
  const int64_t yDims[4] = {N, K, OH, OW}, yStr[4] = {K * OH * OW, OH * OW, OW, 1};
  const int64_t xUid = 0, wUid = 1, yUid = 2;

  cudnnBackendDescriptor_t conv = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv));
  cudnnConvolutionMode_t cmode = CUDNN_CROSS_CORRELATION;
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &cmode));
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt));
  const int64_t spatialDims = 2;
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatialDims));
  const int64_t spatial[2] = {1, 1};
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, spatial));
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, spatial));
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, spatial));
  CK(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, spatial));
  CK(cudnnBackendFinalize(conv));

  cudnnBackendDescriptor_t x = makeTensor(xUid, xDims, xStr, dt);
  cudnnBackendDescriptor_t w = makeTensor(wUid, wDims, wStr, dt);
  cudnnBackendDescriptor_t y = makeTensor(yUid, yDims, yStr, dt);

  cudnnBackendDescriptor_t op = nullptr;
  if (mode == "fwd") {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv);
    float alpha = 1.f, beta = 0.f;
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  } else if (mode == "bwd_data") {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, &op));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, conv);
    float alpha = 1.f, beta = 0.f;
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  } else {
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, &op));
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, y);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, x);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, w);
    setD(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, conv);
    float alpha = 1.f, beta = 0.f;
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    CK(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  }
  CK(cudnnBackendFinalize(op));

  cudnnBackendDescriptor_t graph = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph));
  CK(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CK(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  CK(cudnnBackendFinalize(graph));

  AttrBuf gcount = getAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 8);
  int64_t nEng = gcount.ok ? *gcount.as<int64_t>() : 0;
  const int64_t nEngAll = nEng;
  std::printf("\n== 卷积图 %s [N=%lld C=%lld %lldx%lld -> K=%lld] 的 engine 总数 = %lld\n", mode.c_str(),
              (long long)N, (long long)C, (long long)H, (long long)H, (long long)K, (long long)nEng);

  if (std::getenv("SKIP_ENUM")) nEng = 0;

  struct Row {
    int64_t index;
    std::string num, beh, layout;
    int64_t ws = -1, smem = -1;
    int32_t smTarget = -1;
    bool ok = false;
  };
  std::vector<Row> rows;

  std::printf("\n== 逐个 engine 查询属性（engine 描述符 + engine config 描述符）\n");

  std::printf("%-5s %-38s %-52s %-30s %9s %9s %3s\n", "idx", "numerical_note", "behavior_note",
              "preferred_layout", "workspace", "smem", "sm");
  for (int64_t i = 0; i < nEng; ++i) {
    cudnnBackendDescriptor_t eng = nullptr;
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &eng));
    setD(eng, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, graph);
    CK(cudnnBackendSetAttribute(eng, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &i));
    cudnnStatus_t st = cudnnBackendFinalize(eng);
    if (st != CUDNN_STATUS_SUCCESS) {
      std::printf("%-5lld  finalize 失败: %s\n", (long long)i, cudnnGetErrorString(st));
      cudnnBackendDestroyDescriptor(eng);
      continue;
    }
    Row row;
    row.index = i;
    row.ok = true;
    AttrBuf nn = getAttr(eng, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, 4);
    AttrBuf bn = getAttr(eng, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, 4);
    AttrBuf sm = getAttr(eng, CUDNN_ATTR_ENGINE_SM_COUNT_TARGET, CUDNN_TYPE_INT32, 4);
    row.num = noteList(nn, numNoteName);
    row.beh = noteList(bn, behNoteName);
    if (sm.ok) row.smTarget = *sm.as<int32_t>();

    AttrBuf li = getAttr(eng, CUDNN_ATTR_ENGINE_LAYOUT_INFO, CUDNN_TYPE_BACKEND_DESCRIPTOR, sizeof(cudnnBackendDescriptor_t));
    if (li.ok) {
      cudnnBackendDescriptor_t ldesc = *li.as<cudnnBackendDescriptor_t>();
      AttrBuf uid = getAttr(ldesc, CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID, CUDNN_TYPE_INT64, 8);
      AttrBuf lt = getAttr(ldesc, CUDNN_ATTR_LAYOUT_INFO_TYPES, CUDNN_TYPE_LAYOUT_TYPE, 4);
      if (uid.ok && lt.ok) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "uid%lld:%s", (long long)*uid.as<int64_t>(),
                      layoutName(int(*lt.as<int32_t>())));
        row.layout = buf;
      } else {
        row.layout = "-";
      }
    } else {
      row.layout = "(无)";
    }

    cudnnBackendDescriptor_t cfg = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg);
    setD(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, eng);
    if (cudnnBackendFinalize(cfg) == CUDNN_STATUS_SUCCESS) {
      AttrBuf ws = getAttr(cfg, CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 8);
      AttrBuf sh = getAttr(cfg, CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED, CUDNN_TYPE_INT64, 8);
      if (ws.ok) row.ws = *ws.as<int64_t>();
      if (sh.ok) row.smem = *sh.as<int64_t>();
    }
    cudnnBackendDestroyDescriptor(cfg);

    char wsStr[32], smStr[32];
    std::snprintf(wsStr, sizeof(wsStr), "%lld", (long long)row.ws);
    std::snprintf(smStr, sizeof(smStr), "%lld", (long long)row.smem);
    std::printf("%-5lld %-38s %-52s %-30s %9s %9s %3d\n", (long long)i,
                row.num.empty() ? "-" : row.num.c_str(),
                row.beh.empty() ? "-" : row.beh.c_str(), row.layout.c_str(), wsStr,
                smStr, row.smTarget);
    rows.push_back(row);
    cudnnBackendDestroyDescriptor(eng);
  }

  cudnnBackendDescriptor_t heur = nullptr;
  CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
  setD(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
  cudnnBackendHeurMode_t hmode = CUDNN_HEUR_MODE_A;
  if (const char* e = std::getenv("HEUR_MODE"))
    hmode = cudnnBackendHeurMode_t(std::atoi(e));
  std::printf("\n== heuristic（mode %d）返回的候选 engine config ==\n", (int)hmode);
  CK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hmode));
  CK(cudnnBackendFinalize(heur));
  int64_t nCfg = 0;
  CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                              CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &nCfg, nullptr));
  std::vector<cudnnBackendDescriptor_t> cfgs(size_t(nCfg), nullptr);
  for (auto& c : cfgs) CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &c));
  int64_t got = 0;
  CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                              CUDNN_TYPE_BACKEND_DESCRIPTOR, nCfg, &got, cfgs.data()));
  std::printf("候选数 = %lld（count 查询报 %lld，实际取回 %lld）\n", (long long)got, (long long)nCfg, (long long)got);
  for (int64_t i = 0; i < got; ++i) {
    cudnnBackendDescriptor_t plan = nullptr;
    CK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
    CK(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    setD(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, cfgs[i]);
    cudnnStatus_t sp = cudnnBackendFinalize(plan);
    if (sp != CUDNN_STATUS_SUCCESS) {
      std::printf("  #%2lld  plan finalize 失败: %s\n", (long long)i, cudnnGetErrorString(sp));
      cudnnBackendDestroyDescriptor(plan);
      continue;
    }
    AttrBuf ws = getAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 8);
    AttrBuf js = getAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 1);
    std::string json = js.ok ? std::string(js.as<char>(), size_t(js.count)) : std::string();
    auto field = [&](const char* key) -> std::string {
      size_t p = json.find(key);
      if (p == std::string::npos) return "?";
      p += std::strlen(key);
      size_t q = json.find_first_of(",}\"" , p);
      return json.substr(p, q - p);
    };
    std::string knobs = json;
    size_t kp = knobs.find("\"knobChoices\":{");
    if (kp != std::string::npos) {
      kp += 15;
      size_t kq = knobs.find('}', kp);
      knobs = knobs.substr(kp, kq - kp);
    } else {
      knobs = "-";
    }
    std::printf("  #%2lld  engineId=%-4s workspace=%-9lld smVersion=%-5s  旋钮={%s}\n", (long long)i,
                field("\"engineId\":").c_str(), ws.ok ? (long long)*ws.as<int64_t>() : -1LL,
                field("\"smVersion\":").c_str(), knobs.c_str());
    std::ofstream fo(("candidate_json_" + mode + "_" + std::to_string(i) + ".json").c_str());
    fo << json;
    cudnnBackendDestroyDescriptor(plan);
  }

  std::printf("\n== 汇总 ==\n");
  int nRuntime = 0, nTensorCore = 0, nWinograd = 0, nFFT = 0, nNonDet = 0, nNoKnobFinalize = 0;
  for (const Row& r : rows) {
    if (r.beh.find("RUNTIME_COMPILATION") != std::string::npos) ++nRuntime;
    if (r.num.find("TENSOR_CORE") != std::string::npos) ++nTensorCore;
    if (r.num.find("WINOGRAD") != std::string::npos) ++nWinograd;
    if (r.num.find("FFT") != std::string::npos) ++nFFT;
    if (r.num.find("NONDETERMINISTIC") != std::string::npos) ++nNonDet;
    if (r.ws < 0) ++nNoKnobFinalize;
  }
  std::printf("  全局 engine=%lld，本机能 finalize 的=%zu；其中 runtime 编译=%d  TensorCore=%d  "
              "Winograd=%d  FFT=%d  非确定性=%d\n",
              (long long)nEngAll, rows.size(), nRuntime, nTensorCore, nWinograd, nFFT, nNonDet);
  std::printf("  无法只靠 engine 描述符 finalize 成 config 的（缺省 knob 不完整）=%d\n", nNoKnobFinalize);

  return 0;
}
