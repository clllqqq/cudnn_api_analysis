# PyTorch → cuDNN backend API 调用链分析

> 分析对象：本机 `venvs/torch` 中的 PyTorch 2.14.0+cu130（CUDA 13.0 / cuDNN 9.24.0），GPU 为 RTX 5060 Laptop (sm_120)。
> 结论基于 **静态符号表 + 源码归属 + 运行时 cuDNN 调试日志** 三条独立证据交叉验证，全部命令可复现（见 §7）。

---

## 1. 一页速览

| 问题 | 结论 |
| --- | --- |
| PyTorch 链接哪些 cuDNN 库 | 只有 `libcudnn.so.9`（分发层），实现在 `libcudnn_graph/ops/adv/cnn.so.9` |
| 引用了多少 cuDNN 导出符号 | **83 个**（清单见 `cudnn_imported_symbols.txt`） |
| 入口分几类 | **2 类**：Graph/backend-descriptor API 6 个；传统(legacy)算子 API 66 个；句柄·描述符·工具 11 个 |
| 卷积前反向实际走哪条 | **Graph API**（cuDNN frontend，header-only 内联），legacy `cudnnConvolutionForward` 只在 v8 路径关闭时兜底 |
| SDPA(cuDNN attention) | **Graph API**，无 legacy 调用 |
| BatchNorm / RNN / CTC / int8 Pooling | **legacy API**（`*_v8` / `*Ex` 变体） |
| `F.grid_sample` | **完全不调用 cuDNN**（2.x 已移除）；只有私有 op `aten.cudnn_grid_sampler` 会走 `cudnnSpatialTfSamplerForward` |
| legacy 卷积内部是否走图机制 | **是**：内部走 `cudnn::backend::execute`（日志里的 `cudnnBackendExecuteInternal`），但**不经过**公开的 `cudnnBackend*` 入口，详见 §5.3 |
| 日志解析最大的坑 | `cudnnGraphLibraryConfigInit`、`cudnnBackendExecuteInternal`、`cudnnCTCLossInternal` 等是 **cuDNN 内部函数**，不是 PyTorch 发起的调用 |
| `cudnnCreate` 的 handle 是什么 | 进程内不透明的上下文对象（约 2 KB 堆块）：绑定 CUDA 设备 + 当前 stream + 引擎/图库运行时状态；PyTorch 按"每设备每线程"池化复用，并且**从不调用 `cudnnDestroy`**（§10） |
| legacy 路径能不能选引擎 | 能，但只到"算法"这一层：8 个 `cudnnConvolution*Algo_t` 枚举 + 预估/实测排序；engine_id 与旋钮只有 graph API 能控（§11.4） |
| 不用 graph API 能做卷积吗 | 能：13 个 legacy 调用即可完成一次 Conv2d，全程 0 个 `cudnnBackend*`（§5.5）；代价是引擎选择与融合的粒度都粗 |
| cuDNN 的 engine 是什么 | 同一张算子图的一种"kernel 分组 + 旋钮"实现方案；按 GPU 架构预编译或 JIT（引擎记录里带 `smVersion`），**是软件不是硬件单元**（§11.5） |
| `graph` 与三个 engine 库的关系 | **宿主 ↔ 自注册插件**：引擎库 `NEEDED` 依赖 graph，graph 对它们 0 符号引用、只在运行时 `dlopen`，引擎库在静态构造器里用 `LibraryLoader::set*Func` 反向注册（§4.7） |

---

## 2. 分析对象与环境

| 项 | 值 |
| --- | --- |
| torch | `2.14.0+cu130` |
| torch.version.cuda | `13.0` |
| torch.backends.cudnn.version() | `92400`（9.24.0） |
| Python | 3.14（venv：`/home/liquan/venvs/torch`） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 被分析的二进制 | `venvs/torch/lib/python3.14/site-packages/torch/lib/libtorch_cuda.so`（含 `torch/csrc` 与 `ATen` 的全部 CUDA/cuDNN 实现） |
| cuDNN 安装位置（**实际生效**） | wheel `nvidia-cudnn-cu13 9.24.0.43`：库 `/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia/cudnn/lib/`，头 `/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia/cudnn/include/`（`ldd libtorch_cuda.so` 就解析到这里；`torch.backends.cudnn.version()=92400`） |
| cuDNN 安装位置（本机另一份，未生效） | apt：`libcudnn9-{cuda-12,dev-cuda-12,headers-cuda-12} 9.26.0.51-1` → 库 `/usr/lib/x86_64-linux-gnu/libcudnn*.so.9.26.0`，头 `/usr/include/x86_64-linux-gnu/cudnn.h`（9.26.0，CUDA 12 版；`/usr/local/cuda-12.8` 里没有 cuDNN） |

---

## 3. 研究方法

### 3.1 静态：从二进制里"点名"

PyTorch 的 cuDNN 调用点分布在 `libtorch_cuda.so`（PyTorch 自身）和 `cudnnFrontend`（第三方 header-only 库，编译时内联进去，运行期没有独立 `.so` 可以 hook）。因此静态分析取"导入符号"这一层：

```bash
nm -D --undefined-only libtorch_cuda.so | awk '{print $NF}' | grep -i '^cudnn' | sort -u
objdump -p libtorch_cuda.so | grep NEEDED        # 只依赖 libcudnn.so.9
```

得到 83 个未定义符号，即 PyTorch 真正会去解析的 cuDNN 入口全集。符号在 9.x 里带版本后缀（`sym@libcudnn.so.9`），统计时应先剥掉。

### 3.2 源码归属：符号 → 调用点

为回答"哪个算子由哪段代码调用"，按 tag 抓取 PyTorch 源码中与 cuDNN 相关的三个目录（`v2.14.0` 与 `main` 的文件清单一致）：

- `aten/src/ATen/native/cudnn/` —— 算子主体（Conv_v7/Conv_v8/BatchNorm/RNN/MHA/LossCTC/GridSampler/AffineGridGenerator）
- `aten/src/ATen/cudnn/` —— 句柄与描述符封装（Handle.cpp / Descriptors.{h,cpp} / Types）
- `aten/src/ATen/native/quantized/cudnn/` —— int8 量化引擎（Pooling / BinaryOps）

对每个文件做 `grep -ho '\bcudnn[A-Z][A-Za-z0-9_]*'`，把 83 个符号映射回调用点（结果见 §6）。少数符号不在上述目录，用关键字反查补齐，例如：

- `cudnnGetVersion` / `cudnnGetCudartVersion` → `aten/src/ATen/cuda/detail/CUDAHooks.cpp`
- `cudnnGetErrorString` → `aten/src/ATen/cuda/Exceptions.h`（`AT_CUDNN_CHECK` 宏）

### 3.3 动态：cuDNN 调试日志还原真实调用序列

cuDNN 自带日志，会记录每一次公开 API 调用：

```bash
CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=/tmp/cudnn_xxx.log python your_case.py
grep -ho 'function cudnn[A-Za-z0-9_]*' /tmp/cudnn_xxx.log | sort | uniq -c | sort -rn
```

关键点：**frontend 是 header-only，没有可 hook 的运行时库**；但它每次都要落到 backend descriptor API，而后者被日志完整记录 —— 这正是本仓库 `cudnn_frontend_map.py` 的思路（从日志反推 frontend 图 API 的 4 个阶段：建图 → 引擎选择 → 执行计划 → 执行）。

被测算子（`cudnn_per_op.py` 改写版）：Conv2d 前反向、BatchNorm2d、2 层 LSTM、cuDNN SDPA、CTCLoss、`F.grid_sample`，另加两个私有 op。

### 3.4 反例验证

只统计"日志里出现了什么"会漏掉两种情况，所以对每个可疑项都做了条件对比：

| 现象 | 验证方式 | 结论 |
| --- | --- | --- |
| CTC 完全没有日志 | 2D int64 labels → CUDA int32 1D labels → 对照 `LossCTC.cpp` 的判定条件 | 只有满足条件才走 cuDNN；否则走 ATen 自有 kernel |
| `F.grid_sample` 没有日志 | 公开 op vs 私有 op `aten.cudnn_grid_sampler` | 公开 op 已不含 cuDNN 分支，私有 op 仍走 |
| 日志里的函数比导入符号还多 | 与 83 个导入符号求差集 | 多出来的是 cuDNN 内部函数，不是 PyTorch 调用 |

---

## 4. 静态结论：83 个符号的构成

### 4.1 两类入口

1. **Graph / backend-descriptor API（6 个）** —— 由 header-only 的 `cudnn_frontend` 内联调用，PyTorch 侧看到的只是这 6 个 C 函数：

   `cudnnBackendCreateDescriptor`、`cudnnBackendDestroyDescriptor`、`cudnnBackendSetAttribute`、`cudnnBackendGetAttribute`、`cudnnBackendFinalize`、`cudnnBackendExecute`

2. **传统(legacy)算子 API（66 个）** —— 按算子域划分，以 `*_v8` / `*Ex` 为主：

   - 卷积：`cudnnConvolutionForward / BackwardData / BackwardFilter / BiasActivationForward`，算法搜索 `cudnnFind*AlgorithmEx` + `cudnnGet*Algorithm_v7`，workspace 查询，conv/filter 描述符
   - BatchNorm：`cudnnBatchNormalization{ForwardTrainingEx, BackwardEx, ForwardInference}` + 3 个 workspace/reserve 查询
   - RNN：`cudnnRNNForward`、`cudnnRNNBackwardData_v8`、`cudnnRNNBackwardWeights_v8`、`cudnnSetRNNDescriptor_v8`、RNNData/dropout 描述符、`cudnnGetRNN{WeightParams, TempSpaceSizes, WeightSpaceSize}`
   - CTC：`cudnnCTCLoss(_v8)`、`cudnnGetCTCLossWorkspaceSize(_v8)`、`cudnnSetCTCLossDescriptorEx/_v9`
   - SpatialTransformer：`cudnnSpatialTfSampler{Forward,Backward}`、`cudnnSpatialTfGridGenerator{Forward,Backward}`
   - Pooling/Activation 描述符：`cudnnPoolingForward`、`cudnnCreate/SetPooling2dDescriptor`、activation 描述符

3. **句柄 · 张量描述符 · 运行时工具（11 个）**

   `cudnnCreate`、`cudnnSetStream`、`cudnnGetStream`、`cudnnCreate/Set/GetTensorNdDescriptor` 等张量描述符、`cudnnGetVersion`、`cudnnGetCudartVersion`、`cudnnGetErrorString`、`cudnnGetLastErrorString`

### 4.2 值得注意的"缺席"

- **没有** `cudnnSoftmaxForward`、`cudnnActivationForward`、`cudnnDropoutForward`、`cudnnReduceTensor`、`cudnnTransformTensor`、`cudnnAddTensor` —— 这些 PyTorch 全部用自有 CUDA kernel
- **没有** `cudnnDestroy` —— `aten/src/ATen/cudnn/Handle.cpp` 明确不销毁句柄（规避 atexit 期 CUDA 上下文已析构的崩溃）
- **没有** `cudnnGetConvolutionForwardAlgorithmV8` 之类的 graph 版算法查询 —— 引擎选择全部由 frontend 完成

### 4.3 cuDNN 9 的分库结构

PyTorch 只链接 `libcudnn.so.9`（分发/兼容层），同一批符号在子库中同样存在，真实实现分布如下：

| 符号族 | 定义所在子库 |
| --- | --- |
| `cudnnCreate`、`cudnnBackendExecute`（graph API、句柄） | `libcudnn_graph.so.9` |
| `cudnnConvolutionForward`、卷积 legacy | `libcudnn_cnn.so.9` |
| `cudnnBatchNormalizationForwardTrainingEx`、`cudnnPoolingForward`、`cudnnSpatialTfSamplerForward` | `libcudnn_ops.so.9` |
| `cudnnRNNForward`、`cudnnCTCLoss` | `libcudnn_adv.so.9` |

且 `libcudnn_cnn.so.9` 依赖 `libcudnn_graph.so.9`：**cuDNN 9 里 legacy 接口与 graph 接口共用同一套引擎/图机制**。运行时日志也印证了这点——legacy 调用旁边同样会出现内部函数 `cudnnGraphLibraryConfigInit`。但要注意：**PyTorch 侧看到的入口仍是上面两类，legacy 调用不会经过公开的 `cudnnBackend*` 入口**；它走的是同一套机制的内部 C++ 实现，证据见 §5.3。

### 4.4 `ops` / `cnn` / `adv` 三个库之间的依赖关系

**结论：严格单向分层——`graph` 是底座，`ops` 依赖 `graph`，`cnn` 与 `adv` 各自依赖 `ops + graph`；`cnn` 和 `adv` 之间没有任何直接依赖（互不引用、也互不加载）。**

依赖图（`objdump -p` 的 NEEDED + 运行时 `LD_DEBUG=libs` 实测）：

```
libcudnn.so.9  (0.1 MiB)   分发器：dlopen + dlsym，导出 233 个转发跳板
   │ dlopen
   ├─ libcudnn_graph.so.9   (122 MiB)  核心：backend descriptor / 引擎运行时 / 句柄 / 日志工具
   │        ↑ NEEDED
   ├─ libcudnn_ops.so.9     (38 MiB)   张量·滤波·pooling·activation·dropout·BN·SpatialTf 描述符 + BN 执行
   │        ↑ NEEDED            ↑ NEEDED
   ├─ libcudnn_cnn.so.9     (4 MiB)    卷积族（fwd/bwd/bias-act/算法搜索）
   └─ libcudnn_adv.so.9     (104 MiB)  RNN / CTC / MHA
                             （平行旁支：heuristic 58 MiB、engines_precompiled 248 MiB、
                               engines_runtime_compiled 32 MiB、engines_tensor_ir 192 MiB 都只依赖 graph——
                               这四个库是 graph 的"自注册插件"，graph 反向 0 引用、只在运行时 dlopen，见 §4.7；
                               ext 122 MiB 无任何 cuDNN NEEDED 依赖，提供 causal conv 等 subquadratic 扩展）
```

符号级交叉引用（`nm -D -u` ∩ `nm -D --defined-only`，表内为符号个数）：

| 导入方 \ 定义方 | graph | ops | cnn | adv |
| --- | --- | --- | --- | --- |
| graph | — | 0 | 0 | 0 |
| ops | 53 | — | 0 | 0 |
| cnn | 92 | 10 | — | 0 |
| adv | 48 | 4 | 0 | — |

耦合点具体是什么：

- `ops ← graph`（53）：日志工具（`logArg*`、`CallbackFun`、`logEnviron`）与 backend 机制
- `cnn ← graph`（92）：`cudnn::backend::*`（OperationGraph / EngineHeuristics / ExecutionPlan / execute，见 §5.3）
- `cnn ← ops`（10）：包含**公开 API `cudnnBatchNormalizationBackwardEx`**，以及 `cudnn::ops::{create,destroy}TensorDescriptorEx` 等内部函数
- `adv ← ops`（4）：`cudnn::ops::{create,destroy}TensorDescriptorEx`、dropout 的 `logArg`
- **`cnn` ↔ `adv`：0 个符号**，互不依赖

运行时证据（同一台机器，`LD_DEBUG=libs` 看 dlopen 顺序）：

| 程序 | 实际加载的 cuDNN 库 |
| --- | --- |
| legacy 卷积 demo（`cudnnConvolutionForward`） | `libcudnn` + `graph` + `ops` + `cnn` + `heuristic` + 三个 `engines_*` |
| graph demo（只用 `cudnnBackend*`） | `libcudnn` + `graph` + `heuristic` + 三个 `engines_*` —— **完全不加载 ops / cnn / adv** |

也就是说：**graph API 路径不需要这三个算子库**（算子由引擎库实现），只有 legacy 入口才经过 `ops`/`cnn`/`adv`。这与 §5.3 的"两条入口、一个内核"互相印证。

分发器 `libcudnn.so.9` 干什么（反汇编确认）：它导出 233 个公开符号，每个都是跳板——首次调用经 `dlopen` + `dlsym` 找到子库里的真实实现并缓存函数指针，之后直接 `call *%r12` 转发；它自身没有任何 cuDNN 的 NEEDED 依赖。

83 个 PyTorch 符号按实现库分布：`ops` 32、`adv` 20、`cnn` 18、`graph` 13（合计 83；`ext` 一个都没有，PyTorch 不用 causal conv 扩展）。逐符号清单见 `cudnn_symbol_owners.txt`。

### 4.5 为什么 `ops` 依赖 `graph`（以及 `cnn` 为什么更重）

`ops` 从 `graph` 导入 53 个符号，按用途分：

| 类别 | 个数 | 例子 |
| --- | --- | --- |
| 日志/追踪设施 | 35 | `logArg*`（参数打印）、`logFunction`、`logEnviron`、`log{info,warn,err}_dbg`、`traceback_iretf_impl`、`TracebackLogger`、`CallbackFun`、`hasValidLoggingCallback`、`startingTime` |
| 描述符 setter 的实体实现 | 4 | `cudnn::graph::setTensorNdDescriptor` / `setTensorNdDescriptorEx` / `setFilterNdDescriptor` / `setActivationDescriptor` |
| 子库加载器 | 2 | `cudnn::graph::LibraryLoader::{getInstance,loadLibraries}` |
| 运行时杂项 | 12 | `cudnnGetPid/Tid/Time`、`cudnnSnprintf`、`cudnnGetStream`、`cudnn_graph_version`、`userData`、`GPVar::IsFromExternalOverride`、`Fork/JoinPriorityStreams`、`isForwardCompatEnabled` |

四个要点：

1. **公共底座在 `graph`**：句柄、日志/traceback、时间与格式化工具、子库加载器、版本信息都实现在最底层的 graph 里，三个算子库 + engines + heuristic 都复用它。若把这些放到 `ops`，就会形成 `graph → ops → graph` 的循环依赖。
2. **公开 setter 只是薄壳**：`cudnnSetTensor4dDescriptor`、`cudnnSetFilter4dDescriptor`、`cudnnSetActivationDescriptor`、`cudnnSetPooling2dDescriptor`、`cudnnSetDropoutDescriptor` 的符号都定义在 `ops`，但真正干活的是 graph 里的 `cudnn::graph::setTensorNdDescriptor` 等 4 个 C++ 函数（`ops` 导入它们）。`adv` 也用到其中 1 个（`setTensorNdDescriptor`，RNN 数据），`cnn` 一个都不用（它只跟 backend descriptor 打交道）。
3. **`ops` 完全不碰引擎**：它从 graph 导入的 53 个符号里，`cudnn::backend::*`（OperationGraph / EngineHeuristics / ExecutionPlan / execute）**一个都没有**；对比 `cnn` 的 92 个导入里有 35 个是 backend/引擎类。也就是说 **BN / pooling / activation / grid sampler 这些算子是 `ops` 自己的 kernel，不走引擎机制**——日志也印证：`cudnnBackendExecuteInternal` 在 legacy 卷积日志（cnn）里出现 3 次，而在 `fe_batchnorm.log`、`fe_gs.log`、`fe_affine.log` 里都是 **0 次**。
4. 于是 `ops → graph` 成了一条**链接期强依赖**（DT_NEEDED）：只要用了任意一个 legacy 描述符 API，`ops` 和 `graph` 就会被一起加载（`LD_DEBUG=libs` 可验证）；而纯 graph API 的程序根本不加载 `ops`。

### 4.6 三个库分别从 `graph` 导入了什么

逐符号清单见 `cudnn_graph_imports.txt`（`nm -D -u ∩ nm -D --defined-only` + `c++filt`）。按用途归类：

| 类别 | `ops`（53） | `cnn`（92） | `adv`（48） |
| --- | --- | --- | --- |
| 日志/追踪（`logArg*`、`logFunction`、`logEnviron`、`log*_dbg`、`traceback_iretf_impl`、`TracebackLogger`、`CallbackFun`、`startingTime`） | 35 | 33 | 27 |
| **backend / 引擎机制**（`OperationGraph`、`EngineHeuristics`、`EngineConfig`、`ExecutionPlan`、`execute`、`*Operation::finalize_internal`、相关 `vtable for cudnn::backend::*`） | **0** | **30** | **0** |
| 描述符 setter / 描述符实体 | 4 | 7 | 1 |
| 库加载 / 版本 / 句柄 / 运行时工具 | 14 | 12 | 12 |
| 其它（host malloc、nvrtc、`fusion::Graph` …） | 0 | 10 | 8 |

三家的"性格"完全不同：

- **`ops`：只要基础设施 + 描述符实体。** 它导入的 4 个 setter 是 `cudnn::graph::setTensorNdDescriptor / setTensorNdDescriptorEx / setFilterNdDescriptor / setActivationDescriptor`——即公开 API `cudnnSetTensor4dDescriptor` 这类"壳在 ops、芯在 graph"。**没有任何 `cudnn::backend::*`**。
- **`cnn`：唯一把 graph 当执行引擎用的库。** `cudnn::backend::OperationGraph::appendOperation`、`EngineHeuristics::{ctor,setHeuristicsResult,getHeursticsResult}`、`EngineConfig::{ctor,...}`、`ExecutionPlan::{setEngineConfig,finalize_internal}`、`cudnn::backend::execute`，外加 `Convolution{Forward,Dgrad,Wgrad}Operation::finalize_internal`、`PointwiseOperation/PointwiseDescriptor`（bias+activation 融合）和大量 `vtable for cudnn::backend::*`（跨 DSO 构造/析构这些 C++ 对象）。这正是 §5.3"legacy 卷积内部走同一套引擎机制"的符号级证据。
- **`adv`：和 ops 一样不碰引擎**，但额外导入 `nvrtcHelper::load()`（RNN/attention 会现场编译 kernel）、`cudnnHostMalloc/Free/Calloc`，以及 `cudnn::ops::{Fork,Join}{Priority,Internal}Streams` 和 `cudnn::ops::{Get,Set}Stream`（多流调度/流管理工具）。

两个容易误解的点：

1. **命名空间 ≠ 实现所在库**：`cudnn::ops::ForkPriorityStreams`、`cudnn::ops::isTF32Enabled`、`cudnn::ops::GetStream/SetStream` 名字在 `cudnn::ops` 下，但实现都在 `graph`（由 ops/adv 反过来导入）。
2. 三家共同导入的还有：`userData`、`cudnnGetPid/Tid/Time`、`cudnnSnprintf`、`cudnn_graph_version`、`GPVar`、`cudnn::graph::LibraryLoader` —— 也就是说 `graph` 同时承担了"句柄 + 日志 + 库加载 + 引擎运行时"四种角色。

### 4.7 `graph` 与三个 engine 库的依赖关系

**结论：依赖严格单向——`graph` 是宿主/运行时，三个 engine 库（+ `heuristic`）是自注册插件。引擎库在链接期依赖 `graph`（`NEEDED` 只有 `libcudnn_graph.so.9` + 系统库）；`graph` 对引擎库一个符号都不引用，只在运行时 `dlopen` 它们，引擎库在自己的静态构造器里把入口反向注册进 graph。**

方向与证据（逐符号清单见 `cudnn_engine_imports.txt`）：

| 方向 | 结论 | 证据 |
| --- | --- | --- |
| 引擎库 → `graph` | **强依赖（链接期）** | `objdump -p`：四个库的 `NEEDED` 只有 `libcudnn_graph.so.9` + `z/dl/pthread/rt/stdc++/m/gcc/c`；没有任何 `ops/cnn/adv` |
| `graph` → 引擎库 | **符号级 0 引用** | `libcudnn_graph.so.9` 的未定义符号里，指向引擎库符号、`dummy*Symbol`、`registerRuntimeCompiled` 的个数 = **0** |
| `graph` → 引擎库 | **运行时 `dlopen`（弱依赖）** | `libcudnn.so.9` 和 `libcudnn_graph.so.9` 自身**没有任何 cuDNN 的 `NEEDED`**；graph 里存着 `libcudnn_engines_precompiled.so` / `..._runtime_compiled.so` / `..._tensor_ir.so` / `libcudnn_heuristic.so` 四个文件名字符串，由 `LibraryLoader::loadLibraries()` 按需加载 |
| 加载时机 | **惰性**，不是启动时 | `LD_DEBUG=libs`（`graph_conv/ld_libs.log`）：`initialize program` 在第 146 行，四个库的 `trying file=` 在第 508/560/611/662 行，每次 dlopen 后紧跟一条 `calling init`（静态构造器） |

**引擎库是"极薄的插件"**：对外导出的动态符号几乎为零，内容是内核实现数据 + 静态注册器。

| 库 | 除 SONAME 外导出的动态符号 | 段大小（`.text` / `.nv_fatbin`） |
| --- | --- | --- |
| `libcudnn_engines_precompiled.so.9` | `dummyBackendEnginesPreCompiledSymbol` | 51.5 MiB / 155.1 MiB（预编译 SASS 镜像） |
| `libcudnn_engines_runtime_compiled.so.9` | `dummyBackendEnginesRuntimeCompiledSymbol`、`registerRuntimeCompiled`、8 个 `cudnn::fusion::*CustomLut*` | 11.5 MiB / 3.3 MiB |
| `libcudnn_engines_tensor_ir.so.9` | `dummyBackendEnginesTensorIRSymbol` | 159.5 MiB / 0 |
| `libcudnn_heuristic.so.9` | `dummyHeuristicSymbol` | 2.4 MiB / 0 |

`dummy*Symbol` 不带功能，作用是**ABI/版本标记**（它们带 `@@libcudnn_engines_*.so.9` 版本后缀，且只存在于自己库里，dispatcher 和 graph 都不引用），真正功能走注册回调。

**注册钩子**：四个库都从 graph 导入 `cudnn::graph::LibraryLoader::set*` 系列函数（`cudnn_engine_imports.txt` 关键符号小节）：

| 钩子 | precompiled | runtime_compiled | tensor_ir | heuristic |
| --- | --- | --- | --- | --- |
| `setInitFunc(module, fn)` | 1 | 1 | 1 | 1 |
| `setInstantiator(module, fn)` | 1 | 1 | 1 | — |
| `setHeurFunc` / `setKTMFunc` | — | 1 / 1 | — | 1 / 1 |
| `setAutoHeurGroupFromOpsetFunc` | — | — | — | 1 |
| `loadLibraries()` | — | 1 | — | 1 |

机制：graph 里的 `LibraryLoader` 单例维护一张注册表；四个库各自带 `INIT_ARRAY`（静态构造器），在自己被 `dlopen` 时调用 `setInitFunc/setInstantiator/...` 把"我是哪个模块 + 初始化/实例化/启发式函数在哪"登记进去；`loadLibraries_()` 之后从表里取函数指针回调（反汇编可见 `call *%r13`，参数为固定 flags）。也就是说 graph 不需要 `dlsym` 引擎的 `dummy*Symbol`——**是插件自己把自己挂上去的**（Linux 内核模块式的"自注册插件"模型）。

**每个引擎从 graph 要什么**（符号数，按符号自身命名空间归类，`typeinfo/vtable` 计入其类）：

| 引擎 | 合计 | `cnn::` | `fusion::` | `backend::` | `ops::` | `graph::` | 日志/追踪 | 其它 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `engines_precompiled` | 83 | 27 | 27 | 6 | 4 | 7 | 4 | 8 |
| `engines_runtime_compiled` | 192 | 9 | 51 | 66 | 2 | 12 | 12 | 40 |
| `engines_tensor_ir` | 39 | 8 | 8 | 9 | 1 | 3 | 3 | 7 |
| `heuristic` | 28 | 1 | 2 | 14 | 1 | 6 | 3 | 1 |

读法：

- **整套 C++ 引擎框架定义在 graph，不在引擎库里**：`cudnn::backend::{Engine, OperationGraph, EngineHeuristics, EngineConfig, ExecutionPlan}` 的 vtable 与大部分成员函数（如 `EngineInterface::execute(VariantPack const&, CUstream*)`、`ConvolutionEngine::ConvolutionEngine(...)`、`GenericConvolution::init(...)`）都定义在 `libcudnn_graph.so.9`，由引擎库反过来导入。引擎库负责的是 kernel 本体、旋钮解读和自己那份候选表。
- `engines_precompiled` 主要导入 `cudnn::cnn::`（27）与 `cudnn::fusion::`（27，含 `Graph/Node/PortAlloc` 与 RGCN 等 pattern matcher 的 vtable）；`runtime_compiled` 规模最大（192），额外导入 `nvrtcHelper::load()`（NVRTC 现场编译）和 cublasLt 版本钩子，它也是唯一导出 `registerRuntimeCompiled` 的库；`tensor_ir` 额外导入 `__nvrtcNvvmHandle`（graph 里定义的 NVVM 执行句柄变量）。
- `heuristic` 导入 `EngineHeuristics::addEngineConfig`、`EngineConfig::finalize_internal`、`ExecutionPlan::finalize_internal`、`OperationGraph::{getHeurFeatures,isLargeTensorConv}`——**启发式只负责"挑"，计划的构造与收尾仍回到 graph**。

一句话运行链路：

```
cudnnConvolutionForward / cudnnBackendExecute
        └─ libcudnn_graph.so.9        宿主：OperationGraph → EngineHeuristics → ExecutionPlan → execute
              ├─ dlopen libcudnn_heuristic.so.9    → 注册 setHeurFunc/setKTMFunc，提供候选
              ├─ dlopen libcudnn_engines_*.so.9    → 注册 setInitFunc/setInstantiator，提供 kernel
              └─ 全部执行落到 graph 里的 cudnn::backend::execute（§5.3、§11.8）
```

这也解释了 §4.4 的运行时现象：纯 graph API 的程序**不加载** `ops/cnn/adv` 照样能跑卷积——kernel 在引擎库里、执行框架在 graph 里，两者靠注册表在运行时接上；而 legacy 入口之所以额外拉进 `ops/cnn/adv`，是因为那些库里的公开 API 壳子也在用同一套 graph 引擎机制（§4.5、§5.3）。

### 4.7.1 附：`libcudnn_heuristic.so.9` 与 graph 的相互引用

**一句话：符号依赖与所有引擎库完全同构——heuristic → graph 28 个导入，graph → heuristic 0 引用；区别在角色：heuristic 不提供 kernel，它在静态初始化时把"启发式打分函数"注册进 graph，由 graph（和 `cnn`）按需取用。**

| 方向 | 结论 | 证据 |
| --- | --- | --- |
| heuristic → graph | **28 个符号导入**（见下表） | `nm -D -u` ∩ graph 定义符号；`NEEDED` 只有 `libcudnn_graph.so.9` + 系统库 |
| graph → heuristic | **0 个符号、0 个 `NEEDED`** | graph 未定义符号里匹配 `heur`/`KTM` 的个数 = 0 |
| graph → heuristic | 按文件名 `dlopen` | `libcudnn_graph.so.9` 内含字符串 `libcudnn_heuristic.so`（偏移 98278411）；而 dispatcher `libcudnn.so.9` 只认识 graph/ops/cnn/adv/ext 五个名字，**不知道 heuristic 的存在** |
| 加载时机 | 首个 cuDNN 调用时惰性加载 | `graph_conv/ld_libs.log`：`initialize program`（146 行）之后，第 560 行 dlopen heuristic、第 607 行 `calling init`；顺序是 precompiled(508) → heuristic(560) → runtime_compiled(611) → tensor_ir(662) |
| heuristic 对外 | 只导出 `dummyHeuristicSymbol` | `nm -D --defined-only` 共 2 个符号（另一个是 SONAME 绝对符号），其余全为隐藏符号 |

28 个导入分三类（完整清单见 `cudnn_engine_imports.txt`）：

| 用途 | 个数 | 符号 |
| --- | --- | --- |
| 注册钩子（写回 graph 注册表） | 6 | `LibraryLoader::{setInitFunc, setHeurFunc, setKTMFunc, setAutoHeurGroupFromOpsetFunc, loadLibraries}`、`LibraryConfig::init` |
| 写回 graph 框架对象的操作 | 14 | `EngineConfig::{ctor, finalize_internal}`、`EngineHeuristics::{addEngineConfig, getDeviceProp, getDevicePropertyNonNull}`、`ExecutionPlan::{finalize_internal, get_metadata_internal}`、`OperationGraph::{getHeurFeatures, isLargeTensorConv}`、`cudnnBackendGetMetadataFromGraph`、`vtable for cudnn::backend::{Engine, EngineConfig, ExecutionPlan, OperationGraph}` |
| 辅助 | 8 | `cnn::GenericConvolution::getHeurBFeatures`、`fusion::Graph` 的拷贝构造/析构、`ops::isForwardCompatEnabled`、3 个 `traceback_iretf_impl`、`GPVar::GPVar` |

**注册握手（反汇编实证，`objdump -d libcudnn_heuristic.so.9`）**：

```
heuristic 静态初始化（std::call_once / pthread_once，加载时解析 __once_callable）
   └─ 0x1fe250: LibraryLoader::setInitFunc(module=2, fn=0x1fe270)     ← "我的 init 在哪"
graph LibraryLoader::loadLibraries_()
   └─ mov $0x168f0,%edi ; call *%r13                                  ← 回调插件 init；0x168f0 = 92400 = cuDNN 版本号
        heuristic init 0x1fe270
           ├─ 参数 != 92400 → 直接返回内部状态码 1002（版本不一致，拒绝注册）
           └─ 参数 == 92400 →
                setHeurFunc(type=3, fn) → setHeurFunc(type=4, fn)
                → setAutoHeurGroupFromOpsetFunc(fn) → setHeurFunc(type=6, fn)
                → setKTMFunc(KTMBackend{1}, fn)   ← 3 类启发式 + 1 类 KTM 预测入口
```

即 graph 与插件之间用 **cuDNN 版本号**（本机 92400）做握手校验：graph 只在版本匹配时才调用插件的 init，插件也只在参数等于自己编译时的版本号时才注册。

**谁取用它**：heuristic 只注册、不调用；`getHeurFunc / getKTMFunc / getAutoHeurGroupFromOpsetFunc` 这些 getter 定义在 graph。动态符号表里能看到的取用方是 `cnn`（导入 `getHeurFunc`，legacy 卷积选算法时用）和 `engines_runtime_compiled`（导入 `getTensorIRHeurFunc`）。

**角色总结**：heuristic 不产出 kernel，只回答"这张图上有哪些候选 engine config、哪个值得选"；候选的构造（`EngineConfig` 构造/finalize）与执行计划的收尾（`ExecutionPlan::finalize_internal`）都回到 graph 完成。控制流是双向的（graph 拉起并取用 ↔ heuristic 注册），符号依赖仍然单向。

### 4.7.2 graph 不引用引擎符号，那 kernel 是谁发射的？

**答案分两段：把控制流送过去靠"注册的函数指针"（数据，不是符号）；真正 `cuLaunchKernel` 的是引擎库自己——它自己 `dlsym` 出 CUDA driver API 并自己发射。**

**第一段：指针与虚表都是数据，不需要符号引用**

```
graph 的 LibraryLoader 注册表（单例内的函数指针表）
  ├─ dlopen(libcudnn_engines_*.so)
  │     └─ 插件静态构造器: setInitFunc(mod, &plugin_init) / setInstantiator(mod, &plugin_inst)
  ├─ loadLibraries_(): mov $0x168f0,%edi ; call *%r13      ← 间接调用寄存器里的指针
  └─ 执行期：graph 持有的 Engine/EngineInterface 对象，虚函数经对象内 vtable 指针调用（同样是数据）
```

关键区分：**链接期符号解析 ≠ 运行期函数调用**。插件把函数地址"塞"进 graph 的注册表，graph 用 `call *%r13` 这类间接调用触发——ELF 里不需要任何指向引擎符号的重定位项，所以 `nm -D -u libcudnn_graph.so.9` 里引擎相关符号为 0。这是所有插件的通用做法（Linux 内核模块、COM、dlopen 插件）：宿主定义接口 + 注册表，插件 push 指针，宿主 call 指针。

**第二段：kernel 由引擎库自己发射（实测）**

用 `LD_PRELOAD` 包装 `dlsym()`、在钩子里 `dladdr(__builtin_return_address(0))` 记录调用方 DSO（探针与日志见 `dlsym_probe/`，两个 demo 复现同一结论）：

| DSO | 它自己解析（dlsym）的 CUDA driver 入口 |
| --- | --- |
| `libcudnn_engines_precompiled.so.9` | `cuInit`、`cuLinkCreate_v2`/`cuLinkAddData_v2`/`cuLinkComplete`、`cuModuleLoadData`/`Ex`/`FatBinary`、`cuModuleGetFunction`、`cuFuncSetAttribute`/`GetAttribute`、**`cuLaunchKernel`、`cuLaunchKernelEx`**、`cuModuleUnload`、`cuCtxPushCurrent`/`PopCurrent`、`cuTensorMapEncodeTiled` … |
| `libcudnn_graph.so.9` | `cuDriverGetVersion`、`cuDeviceGet`/`GetAttribute`、`cuGetProcAddress_v2`、`cuModuleLoadData`、`cuModuleGetFunction`、`cuFuncSetAttribute`、`cuOccupancy*`（3 个）——**没有 `cuLaunchKernel`/`cuLink*`** |

注意：只包装 `cuLaunchKernel` 是抓不到的——各 DSO 是对 `libcuda.so.1` 的**句柄**做 `dlsym`，不走全局符号查找，所以必须从 `dlsym` 入口拦。

**于是实际的数据流是**：

```
graph: 建图 → 选引擎 → plan（算 workspace）→ 把 variant pack（uid→指针）+ stream 交给插件
                                                        │  （函数指针 / 虚表，纯数据）
引擎库: 自己的 .nv_fatbin / IR ──cuLink*──> cuModuleLoadData ──> cuModuleGetFunction
        ──cuFuncSetAttribute──> cuLaunchKernel(handle 的 stream, 参数)
```

graph 的角色是**编排与记账**（图、引擎候选、计划、workspace、stream），kernel 怎么编译、怎么发射是引擎库自己的事——这也解释了两边的 driver 入口数量差：graph 只解析 10 个 driver 符号（装载/查询/容量），`engines_precompiled` 解析 33 个（含 JIT 链接与发射）。

顺带回答另一半疑问：`setInstantiator(module, cudnnStatus_t (*)(cudnn::backend::Engine&, bool))` 这个签名说明 graph 会把**已经构造好的引擎对象**（框架类定义在 graph）交给插件，由插件把自己的 kernel 资源与它绑定；执行时 graph 调框架的 `EngineInterface::execute`，其中"发射"这一步落到插件提供的代码——即上面实测到的那些 `cuLaunchKernel` 调用。

### 4.7.3 怎么观察 engine 到底怎么发射 kernel（工具与实测）

三条路都试过（细节与复现命令见 `launch_probe/README.txt`）：

| 手段 | 原理 | 本机结果 |
| --- | --- | --- |
| 驱动符号替换 `launch_probe/launch_trace.c` | `LD_PRELOAD` 包装 `dlsym()`，把引擎取到的 `cuLaunchKernel` 换成自己的包装 | 机制本身可用，但**看不到引擎的发射**：引擎实际是用 `cuGetProcAddress` 取发射入口；而 WSL 版 libcuda 一旦该符号被抢占就会与 preload 互相递归（ping-pong），不能包装 |
| Nsight Systems `nsys --trace=cuda` | CUPTI 注入 | WSL 下拿不到数据（`does not contain CUDA trace data`） |
| **CUPTI / torch.profiler（推荐，可用）** | Kineto 走 CUPTI 记录每次 kernel launch | 完整拿到 **kernel 名、grid/block、动态 smem、寄存器数、stream、耗时** |

工具：`launch_probe/kernel_trace.py`

```bash
/home/liquan/venvs/torch/bin/python launch_probe/kernel_trace.py conv batchnorm rnn attention_forced
# 输出 launch_probe/kernels_<case>.txt（表格）与 kernels_<case>.json（可拖进 chrome://tracing 看时间线）
```

脚本会拿 kernel 名字去各个 cuDNN 子库里做字符串搜索，自动标注这个 kernel 属于哪个库（JIT 生成的 kernel 会标成"runtime 现场编译"）。

**engine_id ↔ 真正的 kernel**（同一次运行里同时开 cuDNN level-3 日志与 CUPTI；证据 `launch_probe/engine_kernel_map.txt`、`kernels_conv_fwd.txt`、`kernels_conv_bwd.txt`）：

| 阶段 | engine_id | 实际 kernel（launch 配置） | 所在库 |
| --- | --- | --- | --- |
| fwd | 28 | `implicit_convolve_sgemm<float,float,int,1024,5,5,3,3,3,1,...>`，grid=[512,1,1] block=[8,8,1] smem=2304 regs=63 | `engines_precompiled` |
| dgrad | 0 | `cudnn::cnn::dgrad2d_grouped_direct_kernel<...>`，grid=[48,1,1] block=[1024,1,1] regs=64 | `engines_precompiled` |
| wgrad | 50 | `nchwToNhwcKernel`×2 → `cutlass__5x_cudnn::Kernel<...ImplicitGemmConvolution...>` grid=[1,1,4] block=[128,1,1] **smem=81920 regs=106** → `reduction` → `nhwcToNchwKernel` | `engines_precompiled` |

其它 case（同方法，`launch_probe/kernels_<case>.txt`）：

| case | kernel（配置） | 说明 |
| --- | --- | --- |
| `batchnorm` | `cudnn::bn_fw_tr_1C11_singleread<float,512,...>` grid=[16,1,1] block=[512,1,1] smem=32912 | legacy BN，kernel 在 `libcudnn_ops`（§4.5） |
| `attention_forced` | `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_1_128x64...` grid=[1,8,2] block=[128,1,1] smem=40960 regs=234 | 名字带 `cudnn_generated`，即 runtime 引擎现场编译出来的 kernel |
| `ctc` / `grid_sample` | 没有任何 cuDNN kernel | 与 §5.1、§5.2 一致 |

由此可以看到"引擎"最具体的样子：**一个 engine = 一组 kernel + 每个 kernel 的 launch 配置**。graph 侧选中的 knob（`TILE_*`、`KERNEL_CFG`、`STREAM_K`…）最终就落到这些 grid/block/动态 smem/寄存器数上；而 `cuLaunchKernel` 的调用点位于引擎库内部（§4.7.2）。

---

> 反过来看"engine 到底是什么、彼此差在哪"，见 §11.9：枚举本机 81/85/75 个 engine、
> heuristic 候选表，以及 17 个候选各自发出的 kernel/精度/耗时实测。

## 5. 动态结论：每个算子的实际调用序列

以下均为本机实测（cuDNN 9.24.0），原始日志在 `logs/` 目录。计数只统计**入口记录**（`I!` 且负载不以 `status:` 开头），规则与陷阱见 §8.1。`cudnnCreate` 每次调用会打 2 条入口记录（句柄 + 设备信息），故表中它的数字约为调用数的 2 倍。

| 场景 | 日志 | PyTorch 侧调用（次数） |
| --- | --- | --- |
| Conv2d 前反向 | `fe_conv.log` | `BackendCreateDescriptor`×174、`BackendDestroyDescriptor`×168、`BackendSetAttribute`×108、`BackendGetAttribute`×45、`BackendFinalize`×27、`BackendExecute`×3、`cudnnCreate`×4、`SetStream`×3、`GetVersion`×1（合计 533） |
| BatchNorm2d | `fe_batchnorm.log` | `BatchNormalizationForwardTrainingEx`×1、`GetBatchNormalizationForwardTrainingExWorkspaceSize`×1、`GetBatchNormalizationTrainingExReserveSpaceSize`×1、张量描述符与管理 ×13（合计 16） |
| LSTM（2 层） | `fe_rnn.log` | `GetRNNWeightParams`×3208、`RNNForward`×1、`SetRNNDescriptor_v8`×3、`GetRNNTempSpaceSizes`×1、`GetRNNWeightSpaceSize`×2、RNNData/dropout/张量描述符与管理 ×157（合计 3372） |
| SDPA（cuDNN attention） | `fe_attn.log` | `BackendCreateDescriptor`×159、`BackendDestroyDescriptor`×147、`BackendGetAttribute`×70、`BackendSetAttribute`×39、`GetVersion`×26、`BackendFinalize`×10、`BackendExecute`×1、`cudnnCreate`×2、`SetStream`×1、`GetStream`×1（合计 456） |
| CTCLoss（int32 1D labels） | `fe_ctc4.log` | `SetCTCLossDescriptor_v9`×1、`CreateCTCLossDescriptor`×1、`DestroyCTCLossDescriptor`×1、张量描述符与管理 ×9（合计 12） |
| `aten.cudnn_grid_sampler`（私有） | `fe_gs.log` | `SpatialTfSamplerForward`×1、`SetSpatialTransformerNdDescriptor`×1、`CreateSpatialTransformerDescriptor`×1、`DestroySpatialTransformerDescriptor`×1、张量描述符与管理 ×10（合计 13） |
| `aten.cudnn_affine_grid_generator`（私有） | `fe_affine.log` | `SpatialTfGridGeneratorForward`×1、`SetSpatialTransformerNdDescriptor`×1、`CreateSpatialTransformerDescriptor`×1、`DestroySpatialTransformerDescriptor`×1、张量描述符与管理 ×4（合计 7） |
| `F.grid_sample` / `F.affine_grid`（公开） | —— | **0 次**（日志文件都不会生成） |

读法提示：

- 卷积一次前反向只有 **3 次 `cudnnBackendExecute`**（fwd / dgrad / wgrad 各一次），却伴随 **174 次描述符创建** —— 这是 frontend 的标准形态：绝大多数开销在建图/引擎选择，"执行"本身只是图提交。
- LSTM 的 `GetRNNWeightParams`×3208 是**权重指针遍历**（每个 RNN layer、每类权重矩阵/偏置各查一次），不是 3208 次算子执行。
- SDPA 的 `GetAttribute`×70 异常高（另有 70 条对应的返回记录，按 §8.1 口径不计入），是因为 attention 图节点/张量多，frontend 需要反复回读属性做校验。

### 5.1 CTC 走 cuDNN 的触发条件（`LossCTC.cpp`）

`_use_cudnn_ctc_loss_tensor` 全部满足才会命中（否则静默走 ATen 实现，日志毫无痕迹）：

- `log_probs`：3D、CUDA、`float32`；`targets`：1D、`int32`、contiguous
- `input_lengths` / `target_lengths`：`int32`，长度相等
- `BLANK == 0`、`ctx.userEnabledCuDNN()`
- 每个 `target_lengths[b] < 256` 且 `target_lengths[b] <= input_lengths[b]`

> 复现坑：`nn.CTCLoss` 默认给的 `targets` 是 2D（`[B, S]`），直接跑永远命中不了 cuDNN 分支；`cudnn_per_op.py` 里的 `ctc` 用例就是这种情况（int64 或 2D 都不行）。且日志里看不到公开的 `cudnnCTCLoss`，只出现 `cudnnCTCLossInternal` / `cudnnGetCTCLossWorkspaceSizeInternal`。

### 5.2 `F.grid_sample` 为什么没有 cuDNN

2.x 的 `aten/src/ATen/native/cuda/GridSampler.cu` 已不含任何 cuDNN 引用（`v2.14.0` 实测 0 处），公开 op 全部走自有 kernel。cuDNN 路径仍编译在库里，但只通过私有 op 暴露：`aten::cudnn_grid_sampler`、`aten::cudnn_grid_sampler_backward`、`aten::cudnn_affine_grid_generator`。因此静态符号表里能看到 `cudnnSpatialTf*`，日常却永远不会触发。

### 5.3 `cudnnConvolutionForward` / `cudnnCreateConvolutionDescriptor` 会不会调用 graph API？

**结论：不经过公开的 graph C 入口，但内部复用同一套 backend 图/引擎机制——"两条入口，一个内核"。**

证据一（符号层）：`libcudnn_cnn.so.9` 是实现 `cudnnConvolutionForward` 的库，它对公开 C API `cudnnBackend*` 的未定义引用为 **0**；它从 `libcudnn_graph.so.9` 导入的是 backend 层的 **C++ 内部符号**：

`cudnn::backend::OperationGraph`（`appendOperation`）、`EngineHeuristics`（`setHeuristicsResult`）、`EngineConfig`、`ExecutionPlan`（`setEngineConfig` / `finalize_internal`）、`Engine::get_internal`、`ConvolutionForward/Dgrad/WgradOperation`、`PointwiseOperation/Descriptor`（bias+activation 融合）、`Tensor::finalize_internal`，以及执行入口 **`cudnn::backend::execute(cudnnContext*, const ExecutionPlan&, VariantPack&)`**。

证据二（运行时）：强制 legacy 路径（`TORCH_CUDNN_V8_API_DISABLED=1`）跑一次 Conv2d 前反向，PyTorch 侧只调 legacy 入口（`logs/fe_conv_legacy.log`）：

- 公开 graph 入口 **0 次**：`cudnnBackendCreateDescriptor / SetAttribute / Finalize / Execute / GetAttribute / DestroyDescriptor` 全都没有出现
- 公开 backend 描述符 **0 个**：日志里没有任何 `CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR / OPERATIONGRAPH / ENGINE / EXECUTION_PLAN`（唯一的 `CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR` 来自 handle 的 deviceProp 打印）
- 但每个 legacy 卷积调用内部都落地一次 `cudnnBackendExecuteInternal`，且 1:1 对应：`cudnnConvolutionForward → operation=CONV_FORWARD`、`cudnnConvolutionBackwardData → CONV_BWD_DATA`、`cudnnConvolutionBackwardFilter → CONV_BWD_FILTER`；该内部记录里还带 `engine_id=28`、`knobDesc: type=cudnnBackendDescriptor_t`（内部对象，非 PyTorch 创建的句柄）
- 失败探针的 traceback 里可见 `outputGraph.buildDAG()`、`initGraph(graph)`、`cudnnBackendFinalizeGraphVisualize` —— legacy 内部同样在"建图 + 选引擎"

证据三（描述符）：`cudnnCreateConvolutionDescriptor` 只创建 legacy 的 `cudnnConvolutionStruct`，其后紧跟的只有 `cudnnSetConvolutionNdDescriptor / SetConvolutionGroupCount / SetConvolutionMathType` 三个 legacy setter，**不涉及任何 backend 描述符**（公开的没有，日志可观测的内部也没有）。把它转成图/引擎的动作发生在**执行时**（`cudnnConvolutionForward` 内部），而不是创建描述符时。

| 维度 | 公开 graph API（frontend） | legacy API（Conv_v7 路径） |
| --- | --- | --- |
| PyTorch 调用的入口 | `cudnnBackendCreateDescriptor` / `SetAttribute` / `Finalize` / `Execute` | `cudnnConvolutionForward` / `BackwardData` / `BackwardFilter` |
| 日志中公开 `cudnnBackend*` 入口调用次数 | 533（见 §5 首行） | **0** |
| 公开 backend 描述符（日志中出现次数） | 大量（`CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR` 108、`OPERATIONGRAPH` 60、`ENGINE` 51、`ENGINECFG` 27 …） | **0** |
| 内部是否建图 + 选引擎 | 是（PyTorch 显式建） | 是（cuDNN 内部 `buildDAG` / `initGraph`） |
| 最终执行 | `cudnn::backend::execute`（日志名：`cudnnBackendExecuteInternal`） | 同上 |
| 引擎选择权 | 引擎级：PyTorch 自己过滤候选（`filterEngineConfigs`）、自己计时/缓存计划，还可用 `CUDNN_ATTR_ENGINE_GLOBAL_INDEX` 钉死某个引擎 | 算法级：PyTorch 在 8 个 `cudnnConvolution*Algo_t` 里挑（v7 启发式给列表，或 `benchmark=True` 时用 `Find...Ex` 实测），算法背后的引擎由 cuDNN 内部选（§11.4） |

> 复现：`TORCH_CUDNN_V8_API_DISABLED=1` 关闭 v8 frontend 路径（也可用 `CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=...` 抓日志），脚本见 `probe_legacy_conv.py`，输出见 `logs/fe_conv_legacy.log`。

### 5.4 附：用 graph API 手写一次卷积（demo）

为了验证上面的结论，本仓库带了一个不依赖 PyTorch 的最小 C++ 实现：`graph_conv/graph_conv_backend.cpp`，用 backend descriptor API 完成一次 `1x3x8x8 * 4x3x3x3 (pad=1, stride=1)` 的卷积前向，自带 CPU 参考校验。

```bash
cd graph_conv && ./run_graph_conv.sh      # 编译 + 运行 + 从日志复核调用次数
```

运行结果（`-O2`，RTX 5060，cuDNN 9.24.0）：

```
  heuristics 返回 16 个 engine config（实际取回 15 个，首个句柄=0x...）
  workspace = 0 字节
  y[0][0][0][0]=-0.024430  参考值=-0.024430
  y[0][3][7][7]=0.131271  参考值=0.131271
  最大绝对误差 = 1.192e-07  ->  通过
```

调用规模对比（都是"入口调用"口径，见 §8.1）：

| 入口 | 本 demo（1 次卷积） | PyTorch Conv2d 前反向（3 个卷积 op） |
| --- | --- | --- |
| `cudnnBackendCreateDescriptor` | 25 | 174 |
| `cudnnBackendSetAttribute` | 37 | 108 |
| `cudnnBackendGetAttribute` | 3 | 45 |
| `cudnnBackendFinalize` | 9 | 27 |
| `cudnnBackendExecute` | 1 | 3 |
| `cudnnBackendDestroyDescriptor` | 25 | 168 |

调用序列与 §5 记录的 PyTorch frontend 完全同构（建张量/卷积/算子/图描述符 → heuristics 选引擎 → 执行计划 → variant pack → execute），只是规模小三个数量级。

实现上踩到的四个 cuDNN 9.24 细节（都在 demo 代码里绕过了）：

1. **数组属性只能单次设置**：`CUDNN_ATTR_OPERATIONGRAPH_OPS` / `CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS` 直接一次 `cudnnBackendSetAttribute` 传入数组即可；先传 `NULL` 声明长度的"两步写法"会被拒（`CUDNN_STATUS_BAD_PARAM`，日志里能看到 `E!` 记录）。
2. **heuristics 结果要先自建 engine config 描述符**：查询 `CUDNN_ATTR_ENGINEHEUR_RESULTS` 的元素个数（传 `NULL` 数组）返回 16，但真正取结果时必须先把 16 个 `CUDNN_BACKEND_ENGINECFG_DESCRIPTOR` 创建好再把数组传进去，否则一条都取不到（实测填充 15 个）。
3. **从 heuristics 返回的 engine config 上读不到 engine**：`cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, ...)` 会返回 `CUDNN_STATUS_BAD_PARAM`；PyTorch 的做法是自建 engine cfg 再 `set` 该属性。demo 直接用返回的 config 建执行计划，执行正常。
4. **`CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS` 是必需属性**：卷积描述符漏掉它会 `cudnnBackendFinalize` 失败（`BAD_PARAM`），错误信息不提示原因。

---

### 5.5 附：不用 graph API 手写一次卷积（demo）

和 §5.4 同一个问题（`1x3x8x8 * 4x3x3x3, pad=1, stride=1`），但这次**全程只走 legacy 入口**：`legacy_conv/legacy_conv_backend.cpp`。

调用序列（没有一行 `cudnnBackend*`）：

```
cudnnCreate / cudnnSetStream
cudnnCreateTensorDescriptor ×2 (x, y) + cudnnSetTensor4dDescriptor
cudnnCreateFilterDescriptor + cudnnSetFilter4dDescriptor
cudnnCreateConvolutionDescriptor + cudnnSetConvolution2dDescriptor + cudnnSetConvolutionMathType
cudnnGetConvolutionForwardAlgorithm_v7      <- 挑算法（legacy 的"选引擎"）
cudnnGetConvolutionForwardWorkspaceSize
cudnnConvolutionForward                     <- 真正执行
```

三种挑算法的方式：

```bash
cd legacy_conv && ./run_legacy_conv.sh    # v7 启发式：只取 cuDNN 推荐的第 1 个
./run_legacy_conv.sh --all                # v7 直接要全部 8 个候选
./run_legacy_conv.sh --find               # cudnnFindConvolutionForwardAlgorithmEx 实测排序
```

默认（启发式）运行结果：

```
#1 IMPLICIT_PRECOMP_GEMM  预估 time= -1.0000 ms  workspace=0
y[0][0][0][0]=-0.024430  参考值=-0.024430
y[0][3][7][7]=0.131271  参考值=0.131271
最大绝对误差 = 1.192e-07  ->  通过
```

`--find` 实测（本机 sm_120 + cuDNN 9.24）：

```
DIRECT               不支持（CUDNN_STATUS_NOT_SUPPORTED），跳过
WINOGRAD             不支持（CUDNN_STATUS_NOT_SUPPORTED），跳过
#1 IMPLICIT_GEMM       time=0.0073 ms  ws=0
#2 GEMM                time=0.0770 ms  ws=0
#3 IMPLICIT_PRECOMP_GEMM time=0.0946 ms ws=0
#4 FFT_TILING          time=0.1772 ms  ws=82688 B
#5 WINOGRAD_NONFUSED   time=0.2690 ms  ws=66240 B
#6 FFT                 time=0.6732 ms  ws=34992 B
```

两个一眼可见的事实：

- **启发式与实测会打架**：v7 推荐 `IMPLICIT_PRECOMP_GEMM`（且预估 `time=-1`，cuDNN 9 已不再给估值），实测却是 `IMPLICIT_GEMM` 快约 13 倍。PyTorch 的 `torch.backends.cudnn.benchmark=True` 切到的就是 `Find...Ex` 这条路（§11.4）。
- **老算法会在新架构上消失**：`DIRECT` / `WINOGRAD` 在 sm_120 上直接返回 `NOT_SUPPORTED`，但 8 个枚举仍留在头文件里，`--all` 时 v7 也照样把它们列进候选——所以候选必须按返回状态过滤（demo 与 PyTorch 的 `getValidAlgorithms` 都这么做）。

与 graph demo 的成本对比（均为"入口记录"口径）：

| 入口 | legacy demo | graph demo（§5.4） |
| --- | --- | --- |
| 公开 `cudnnBackend*` 入口 | **0** | 100 |
| 描述符 / 属性调用 | Create 5、Set 6、Destroy 5 | Create 25、Set 37、Destroy 25 |
| 引擎/算法选择 | `GetConvolutionForwardAlgorithm_v7`×1 + `GetConvolutionForwardWorkspaceSize`×1 | `ENGINEHEUR` + `GetAttribute`×3 + `Finalize`×9 |
| 执行 | `cudnnConvolutionForward`×1 | `cudnnBackendExecute`×1 |
| 日志中公开入口总数 | 22 | 103 |

两点容易忽略的细节（都在 `legacy_conv/legacy_conv.log` 里能看到）：

1. PyTorch/demo 侧 `cudnnBackend*` 为 0，但 `cudnnConvolutionForward` 内部仍落地一条 `cudnnBackendExecuteInternal`（`engine_id=0`）——引擎机制照跑，只是选择权在 cuDNN 内部（§5.3）。
2. `cudnnSetTensor4dDescriptor` 内部会调 `cudnnSetTensor4dDescriptorEx`，日志里被记成**独立的一条入口记录**，所以日志计数比程序真实发出的调用多 2 条（§8 的坑）。

### 5.6 全量复跑：`cudnn_per_op.py` 的 7 种情况

用 `run_per_op_all.sh` 把 `/home/liquan/cudnn_per_op.py` 里的每个 case 各跑一遍（每次独立进程、`CUDNN_LOGLEVEL_DBG=3`），日志与 stdout 存在 `logs_per_op/`，每份日志的入口计数汇总在 `logs_per_op/summary.txt`：

| case | 用时（首跑） | 日志 | 公开入口合计 | 主要入口（次数） |
| --- | --- | --- | --- | --- |
| `conv` | 4.8 s | 620 KB | 533 | `BackendCreateDescriptor`×174、`BackendSetAttribute`×108、`BackendGetAttribute`×45、`BackendFinalize`×27、`BackendExecute`×3 |
| `batchnorm` | 2.1 s | 11 KB | 16 | `BatchNormalizationForwardTrainingEx`×1、`GetBatchNormalizationForwardTrainingExWorkspaceSize`×1、`CreateTensorDescriptor`×3 … |
| `rnn` | 1.7 s | 7.2 MB | 3372 | `GetRNNWeightParams`×3208、`CreateTensorDescriptor`×46、`SetRNNDescriptor_v8`×3、`RNNForward`×1 |
| `attention` | 1.4 s | 190 B | **1** | 只有 `GetVersion`×1 —— 见下面说明 |
| `attention_forced` | 1.6 s | 1.2 MB | 456 | `BackendCreateDescriptor`×159、`BackendGetAttribute`×70、`BackendSetAttribute`×39、`BackendFinalize`×10、`BackendExecute`×1 |
| `grid_sample` | 1.4 s | 0 B | **0** | —（§5.2） |
| `ctc` | 1.4 s | 0 B | **0** | —（这个 case 的 labels 是 int64 二维，触发不了 cuDNN，条件见 §5.1） |

（墙钟耗时会随内核缓存/驱动预热波动，同一 case 二次运行通常更快；日志字节数与入口计数是稳定的。）

两个容易误读的点：

1. **`attention` 不等于"走了 cuDNN attention"**：虽然脚本里 `enable_cudnn_sdp(True)`，但默认 SDPA 的后端优先级在这台机器上没落到 cuDNN——日志里只有一条 `cudnnGetVersion`（torch 初始化时问版本），没有任何计算类入口。要强制走 cuDNN 必须用 `attention_forced`（`sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`），它的日志与 `logs/fe_attn.log` 完全一致：`UNIFIED_SDPA_FWD`、runtime 编译引擎（engine_id=8、`smVersion=1200`）。
2. **"0 条日志"是结果不是失败**：`grid_sample` 与 `ctc` 的日志是 0 字节，说明进程根本没加载 cuDNN（延迟加载），这两个 case 的 stdout 仍是 `ok`——脚本把"能否命中 cuDNN"和"程序是否跑通"分开记录（`exit=0` + `ok`）。

```bash
cd /home/liquan/cudnn_api_analysis
./run_per_op_all.sh              # 全部 7 个 case
./run_per_op_all.sh conv rnn     # 只跑指定 case
CUDNN_LOGLEVEL=2 ./run_per_op_all.sh   # 换日志级别（默认 3）
```

case 列表是从 `/home/liquan/cudnn_per_op.py` 里自动提取的（正则匹配 `op == "xxx"`），以后在目标脚本里加分支，直接重跑即可，不需要改这个 runner。

## 6. 附录 A：83 个符号 → 调用点

> 文件名的完整路径：`Conv_v7.cpp`、`Conv_v8.cpp`、`BatchNorm.cpp`、`RNN.cpp`、`LossCTC.cpp`、`GridSampler.cpp`、`AffineGridGenerator.cpp`、`MHA.cpp` 位于 `aten/src/ATen/native/cudnn/`；`Descriptors.h`、`Descriptors.cpp`、`Handle.cpp` 位于 `aten/src/ATen/cudnn/`；`Pooling.cpp`、`BinaryOps.cpp` 位于 `aten/src/ATen/native/quantized/cudnn/`（int8 量化引擎）；另外两个是 `aten/src/ATen/cuda/detail/CUDAHooks.cpp` 与 `aten/src/ATen/cuda/Exceptions.h`。

| 符号 | 调用点 |
| --- | --- |
| `cudnnBackendCreateDescriptor` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBackendDestroyDescriptor` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBackendExecute` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBackendFinalize` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBackendGetAttribute` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBackendSetAttribute` | cudnn_frontend (内联于 Conv_v8.cpp / MHA.cpp / quantized BinaryOps.cpp) |
| `cudnnBatchNormalizationBackwardEx` | BatchNorm.cpp |
| `cudnnBatchNormalizationForwardInference` | BatchNorm.cpp |
| `cudnnBatchNormalizationForwardTrainingEx` | BatchNorm.cpp |
| `cudnnCTCLoss` | LossCTC.cpp |
| `cudnnCTCLoss_v8` | LossCTC.cpp |
| `cudnnConvolutionBackwardData` | Conv_v7.cpp |
| `cudnnConvolutionBackwardFilter` | Conv_v7.cpp |
| `cudnnConvolutionBiasActivationForward` | Conv_v7.cpp |
| `cudnnConvolutionForward` | Conv_v7.cpp |
| `cudnnCreate` | Handle.cpp |
| `cudnnCreateActivationDescriptor` | Descriptors.h |
| `cudnnCreateCTCLossDescriptor` | Descriptors.h |
| `cudnnCreateConvolutionDescriptor` | Descriptors.h |
| `cudnnCreateDropoutDescriptor` | Descriptors.h |
| `cudnnCreateFilterDescriptor` | Descriptors.h |
| `cudnnCreatePoolingDescriptor` | Pooling.cpp |
| `cudnnCreateRNNDataDescriptor` | Descriptors.h |
| `cudnnCreateRNNDescriptor` | Descriptors.h |
| `cudnnCreateSpatialTransformerDescriptor` | Descriptors.h |
| `cudnnCreateTensorDescriptor` | Descriptors.h |
| `cudnnDestroyActivationDescriptor` | Descriptors.h |
| `cudnnDestroyCTCLossDescriptor` | Descriptors.h |
| `cudnnDestroyConvolutionDescriptor` | Descriptors.h |
| `cudnnDestroyDropoutDescriptor` | Descriptors.h |
| `cudnnDestroyFilterDescriptor` | Descriptors.h |
| `cudnnDestroyRNNDataDescriptor` | Descriptors.h |
| `cudnnDestroyRNNDescriptor` | Descriptors.h |
| `cudnnDestroySpatialTransformerDescriptor` | Descriptors.h |
| `cudnnDestroyTensorDescriptor` | Descriptors.h |
| `cudnnDropoutGetStatesSize` | Descriptors.h |
| `cudnnFindConvolutionBackwardDataAlgorithmEx` | Conv_v7.cpp |
| `cudnnFindConvolutionBackwardFilterAlgorithmEx` | Conv_v7.cpp |
| `cudnnFindConvolutionForwardAlgorithmEx` | Conv_v7.cpp |
| `cudnnGetBatchNormalizationBackwardExWorkspaceSize` | BatchNorm.cpp |
| `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize` | BatchNorm.cpp |
| `cudnnGetBatchNormalizationTrainingExReserveSpaceSize` | BatchNorm.cpp |
| `cudnnGetCTCLossWorkspaceSize` | LossCTC.cpp |
| `cudnnGetCTCLossWorkspaceSize_v8` | LossCTC.cpp |
| `cudnnGetConvolutionBackwardDataAlgorithm_v7` | Conv_v7.cpp |
| `cudnnGetConvolutionBackwardDataWorkspaceSize` | Conv_v7.cpp |
| `cudnnGetConvolutionBackwardFilterAlgorithm_v7` | Conv_v7.cpp |
| `cudnnGetConvolutionBackwardFilterWorkspaceSize` | Conv_v7.cpp |
| `cudnnGetConvolutionForwardAlgorithm_v7` | Conv_v7.cpp |
| `cudnnGetConvolutionForwardWorkspaceSize` | Conv_v7.cpp |
| `cudnnGetCudartVersion` | ATen/cuda/detail/CUDAHooks.cpp |
| `cudnnGetErrorString` | ATen/cuda/Exceptions.h |
| `cudnnGetFilterNdDescriptor` | Descriptors.cpp, RNN.cpp |
| `cudnnGetLastErrorString` | (未定位, 错误路径) |
| `cudnnGetRNNTempSpaceSizes` | RNN.cpp |
| `cudnnGetRNNWeightParams` | RNN.cpp |
| `cudnnGetRNNWeightSpaceSize` | RNN.cpp |
| `cudnnGetStream` | (未定位, ATen 工具层) |
| `cudnnGetTensorNdDescriptor` | Descriptors.cpp, RNN.cpp |
| `cudnnGetVersion` | ATen/cuda/detail/CUDAHooks.cpp |
| `cudnnPoolingForward` | Pooling.cpp |
| `cudnnRNNBackwardData_v8` | RNN.cpp |
| `cudnnRNNBackwardWeights_v8` | RNN.cpp |
| `cudnnRNNForward` | RNN.cpp |
| `cudnnRestoreDropoutDescriptor` | Descriptors.h |
| `cudnnSetActivationDescriptor` | Descriptors.h |
| `cudnnSetCTCLossDescriptorEx` | Descriptors.h |
| `cudnnSetCTCLossDescriptor_v9` | Descriptors.h |
| `cudnnSetConvolutionGroupCount` | Descriptors.h |
| `cudnnSetConvolutionMathType` | Conv_v7.cpp, Descriptors.h |
| `cudnnSetConvolutionNdDescriptor` | Descriptors.h |
| `cudnnSetDropoutDescriptor` | Descriptors.h |
| `cudnnSetFilterNdDescriptor` | Descriptors.h |
| `cudnnSetPooling2dDescriptor` | Pooling.cpp |
| `cudnnSetRNNDataDescriptor` | Descriptors.h |
| `cudnnSetRNNDescriptor_v8` | Descriptors.h |
| `cudnnSetSpatialTransformerNdDescriptor` | Descriptors.h |
| `cudnnSetStream` | Handle.cpp |
| `cudnnSetTensorNdDescriptor` | Descriptors.h |
| `cudnnSpatialTfGridGeneratorBackward` | AffineGridGenerator.cpp |
| `cudnnSpatialTfGridGeneratorForward` | AffineGridGenerator.cpp |
| `cudnnSpatialTfSamplerBackward` | GridSampler.cpp |
| `cudnnSpatialTfSamplerForward` | GridSampler.cpp |

---

## 7. 附录 B：复现步骤

```bash
# 0) 环境
TORCH=/home/liquan/venvs/torch/lib/python3.14/site-packages/torch/lib

# 1) 静态：导入符号全集（83 个）
nm -D --undefined-only $TORCH/libtorch_cuda.so | awk '{print $NF}' \
  | grep -i '^cudnn' | sed 's/@.*//' | sort -u > cudnn_imported_symbols.txt

# 2) 静态：依赖与分库
objdump -p $TORCH/libtorch_cuda.so | grep NEEDED          # -> libcudnn.so.9
for s in cudnnConvolutionForward cudnnRNNForward cudnnBackendExecute cudnnBatchNormalizationForwardTrainingEx; do
  for lib in $TORCH/../nvidia/cudnn/lib/libcudnn*.so.9; do
    nm -D --defined-only $lib | grep -qw $s && echo "$s -> $(basename $lib)"
  done
done

# 3) 源码归属（按 tag 抓源码后 grep）
#    https://api.github.com/repos/pytorch/pytorch/contents/aten/src/ATen/native/cudnn?ref=v2.14.0
#    https://api.github.com/repos/pytorch/pytorch/contents/aten/src/ATen/cudnn?ref=v2.14.0
#    https://api.github.com/repos/pytorch/pytorch/contents/aten/src/ATen/native/quantized/cudnn?ref=v2.14.0
for f in *.cpp *.h; do echo "== $f"; grep -ho '\bcudnn[A-Z][A-Za-z0-9_]*' $f | sort -u; done

# 4) 动态：开 cuDNN 日志跑单个算子
CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=/tmp/cudnn_conv.log \
  /home/liquan/venvs/torch/bin/python /home/liquan/cudnn_per_op.py conv
grep -ho 'function cudnn[A-Za-z0-9_]*' /tmp/cudnn_conv.log | sort | uniq -c | sort -rn

# 5) 阶段归类（仓库已有脚本）
/home/liquan/venvs/torch/bin/python /home/liquan/cudnn_frontend_map.py /tmp/cudnn_conv.log

# 6) 两个手写 demo：graph API vs legacy API（同样一次 Conv2d，结果一致）
cd graph_conv  && ./run_graph_conv.sh        # 25 建描述符 / 37 SetAttribute / 1 Execute
cd ../legacy_conv && ./run_legacy_conv.sh    # 0 个 cudnnBackend*；--all / --find 可选
```

日志开关说明：`CUDNN_LOGLEVEL_DBG=3`（最大粒度）、`CUDNN_LOGDEST_DBG=<文件绝对路径|stdout>`。日志是**懒创建**的——如果一个进程从头到尾没调用任何 cuDNN API，日志文件根本不会生成，这本身就是"该算子没走 cuDNN"的判据（本项目里 `ctc` 旧用例、`grid_sample` 都是这样被识别的）。

## 8. 解析日志时的过滤名单（重要）

cuDNN 会把自己内部的函数也打进同一份日志，它们**不是** PyTorch 发起的调用。解析时应排除：

| 内部函数 | 说明 |
| --- | --- |
| `cudnnGraphLibraryConfigInit` | 图库初始化，每个进程/上下文一次 |
| `cudnnBackendExecuteInternal` | 公开 `cudnnBackendExecute` 的内部实现 |
| `cudnnBackendFinalizeGraphVisualize`、`cudnnBackendFinalizeEngineCfgVisualize`、`cudnnBackendExecuteGraphVisualize` | 可视化/调试辅助路径 |
| `cudnnCTCLossInternal`、`cudnnGetCTCLossWorkspaceSizeInternal` | CTC 族：**公开入口在日志里只以 Internal 名字出现** |
| `cudnnSetTensor4dDescriptorEx` | `cudnnSetTensor4dDescriptor` 的内部实现，日志里单独成条（所以一次 set 会看到 `Descriptor` + `DescriptorEx` 两条） |
| `cudnnGetProperty` | cuDNN 属性查询内部实现 |

判定方法很简单：日志里出现的函数名与 `cudnn_imported_symbols.txt` 求差集，差集部分即为内部函数。

### 8.1 计数陷阱：一次调用可能对应多条日志记录

日志记录按级别分三类，**只有第一类才是"一次调用"**：

| 记录类型 | 特征 | 与调用的关系 |
| --- | --- | --- |
| 入口记录 | `I!`，负载是调用参数 | **一次调用 = 一条**，统计就数这个 |
| 返回记录 | `I!`，负载以 `status: type=cudnnStatus_t` 开头 | `cudnnBackend*` 家族每次调用**额外**打一条；legacy 家族一般不打印 |
| 告警/回溯记录 | `W!` / `E!`，负载是 traceback | 附加在**前一次调用**上，函数名沿用同一个 |

三个实例（都来自本仓库日志）：

- `cudnnBackendCreateDescriptor`：入口 + 返回 = 每个调用 2 条 → `fe_conv.log` 里的 348 条对应 **174 次真实调用**（本文档早期版本按 348 计，已修正为 174）
- `cudnnConvolutionForward`（legacy）：1 条入口 + 1 条 `W!` 回溯（内部 `buildDAG` 的失败告警）→ **1 次**调用
- `cudnnCreate`：2 条入口记录（句柄 + 设备信息）→ 计数约为实际调用数的 2 倍

所以 `cudnn_frontend_map.py` 现有的"跳过紧邻同名重复"逻辑不够：成对的入口/返回记录并不总是紧邻（中间可能插入 `cudnnBackendExecuteInternal` 等内部记录）。建议先过滤出**入口记录**（`I!` 且负载不以 `status:` 开头）再统计。

另外注意日志级别：`CUDNN_LOGLEVEL_DBG=2` 只记录告警，**成功的调用完全不出现**（实测 legacy 卷积在 level 2 下只剩 3 条 `_v7` 告警，看不到 `cudnnConvolutionForward`），要完整调用序列必须用 `=3`。

## 9. 局限与未定位项

1. `cudnnGetStream` 的调用者已定位：**vendored cudnn_frontend 的 inline 函数 `cudnn_frontend::getStream()`**。该函数体被内联（`libtorch_cuda.so` 里只剩它的函数内静态变量 `_ZZN14cudnn_frontend9getStreamEvE6stream/outFile`），但同一个二进制里嵌着构建路径字符串 `/__w/pytorch/pytorch/third_party/cudnn_frontend/include/cudnn_frontend/...`，可确认 frontend 是 header-only 编译进 torch 的。它在 SDPA 路径被调用 1 次（`logs/fe_attn.log:20195`，紧跟在 `cudnnGetVersion()` 之后），作用是把 handle 上绑定的 stream 读回来。`cudnnGetLastErrorString` 仍未定位，只在错误路径触发。二者都不影响上面的算子结论。
2. 静态符号表来自本机 wheel（2.14.0+cu130）；`main` 与 `v2.14.0` 的源码文件清单一致，但上游仍在演进（例如 grid_sample 的 cuDNN 分支就是在 2.x 中移除的）。
3. 动态结论限定在本机 cuDNN 9.24.0 + sm_120。引擎可用性受 GPU/cuDNN 版本影响，例如 `MHA.cpp` 里就有按 cuDNN 版本和 SM 版本禁用的 SDPA decode 分支。
4. 只覆盖了日志记录到的 API。cuDNN frontend 内联后若存在不经过 backend descriptor API 的旁路（截至目前未观察到），静态符号表仍是唯一的兜底依据。

## 10. 补充：`cudnnCreate` 返回的 handle 是什么

**一句话**：handle 是 cuDNN 在进程堆上自己分配的一块**不透明上下文对象**（`typedef struct cudnnContext *cudnnHandle_t`，声明见 `nvidia/cudnn/include/cudnn_graph.h:118,123`），里面记录"这次会话绑定的 CUDA 设备 + 当前 stream + 引擎/图库运行时状态"；调用方只拿到一个指针，结构体定义不对外暴露。

### 10.1 里面装了什么（实测）

| 内容 | 证据 |
| --- | --- |
| **绑定的 CUDA stream** | `cudnnSetStream(s)` 之后 `cudnnGetStream` 读回同一个 `s`，且该指针确实写在句柄对象内存里（本机实测偏移 `+0x4b8`）；未设置时 `cudnnGetStream` 返回 NULL |
| **设备上下文**（创建时的当前设备） | 日志里每个 `cudnnHandle_t` 参数都被展开成 `streamId` + `deviceProp`（deviceName / cudaDeviceId / deviceVer=1200 / multiProcessorCount=26 / driverVer / maxSharedMemoryPerBlockOptin / totalGlobalMem），见 `logs/fe_batchnorm.log:22`、`logs/fe_conv.log:28`；`cudnnCreate` 时做一次设备能力探测（`logs/fe_batchnorm.log:14-17`，`compute_capability_major=12 / minor=0`） |
| **引擎/图库的会话状态** | `cudnnCreate` 之后紧跟内部调用 `cudnnGraphLibraryConfigInit`（`logs/fe_batchnorm.log:8`）；graph API 把 handle 挂到图描述符（`CUDNN_ATTR_OPERATIONGRAPH_HANDLE (800)`，`logs/fe_conv.log:1257`）和执行计划（`EXECUTION_PLAN_HANDLE`）上，选引擎/建计划/执行都发生在它标识的会话里 |
| **约 2 KB 的堆对象** | `malloc_usable_size(handle) = 2072` 字节；两次 `cudnnCreate` 返回两个不同地址 |

复现：`/home/liquan/venvs/torch/bin/python probe_handle.py`（输出存于 `logs/handle_probe.out`）：

```
首次 cudnnCreate -> SUCCESS 0x2ebf7290 耗时 504.62 ms
再次 cudnnCreate -> 0x2ed4cab0 耗时 0.612 ms 句柄堆块大小 2072 字节（两次地址不同：True）
未设置 stream 时 cudnnGetStream -> None（NULL）
cudnnSetStream(s1) 后 getStream=0x2ed7ae50，匹配=True，该指针出现在句柄内存偏移 ['0x4b8']
cudnnSetStream(NULL 句柄, 0) -> 2000 (BAD_PARAM)：句柄是几乎所有 API 的必需第一个参数
另一个线程里 cudnnCreate 得到 0x75e88c003f90（与主线程 0x2ebf7290 不同）
```

> 日志里的两副面孔：`cudnnCreate` 会打 **2 条入口记录**——第一条是调用方传出的指针地址（`handle: location=host; addr=0x...`），第二条是设备能力探测结果（`compute_capability_major/minor ... status=SUCCESS`）。这正是 §5 表里 `cudnnCreate` 计数约为调用数 2 倍、以及 §8.1 计数陷阱的来源。

### 10.2 开销与 PyTorch 的对策

| 行为 | 实测 / 源码依据 |
| --- | --- |
| 首次 `cudnnCreate` | 0.2–0.5 s（含 cuDNN 各子库/引擎库的装载与初始化） |
| 之后每次 `cudnnCreate` | ~0.6 ms |
| `cudnnDestroy` | 头文件明确标注 "Calls cudaDeviceSynchronize"，销毁代价高；PyTorch **不调用** |

PyTorch 侧（`aten/src/ATen/cudnn/Handle.cpp` + `aten/src/ATen/cuda/detail/DeviceThreadHandles.h`）：

1. **池化**：`DeviceThreadHandlePool` —— 每个设备一个 handle 池，每个线程一个 `PoolWindow`；线程首次取用时 `reserve()`（池里有空闲就复用，没有才新建），线程退出后归还，池的上界是并发线程数高水位。目的就是躲开创建开销（DataParallel 每个 forward 都会新建一批线程）。
2. **每次取用都重设 stream**：`getCudnnHandle()` 里 `cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream())` —— handle 会记住上一次的 stream，必须让它跟随当前 CUDA stream。
3. **永不销毁**：`destroyCuDNNHandle` 是空函数（源码注释：atexit 阶段 CUDA 上下文可能已经析构，此时调 `cudnnDestroy` 会崩）。

### 10.3 三条使用约束

1. handle 是几乎所有 cuDNN API 的**第一个参数**；传 NULL 直接 `CUDNN_STATUS_BAD_PARAM (2000)`（实测）。
2. handle 与**设备**绑定，cuDNN 官方建议**不要跨线程共享**（`DeviceThreadHandles.h` 开头注释："tied to device ... recommends not to share handles across host threads"）—— 这正是 PyTorch 按 (设备, 线程) 池化的原因。
3. handle ≠ 描述符：tensor/conv/graph/plan 等 `cudnnBackendDescriptor_t` 都是各自独立的对象，handle 只是会话上下文。graph API 里要显式把它设进 `OPERATIONGRAPH_HANDLE`，`cudnnBackendExecute(handle, plan, variantPack)` 再作为第一个参数传一次。

## 11. 补充：graph API 是什么、优势在哪

### 11.1 是什么

graph API（cuDNN backend descriptor API）把一次计算描述成**算子 DAG**：

- 节点 = 算子描述符（`CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`、矩阵乘、pointwise、softmax、attention …）；
- 边 = 虚拟张量（`CUDNN_BACKEND_TENSOR_DESCRIPTOR` 用 `uid` 把算子连起来，真实数据指针留到执行前才由 variant pack 绑定）。

一次执行分四个阶段（`graph_conv/graph_conv_backend.cpp` 与 PyTorch 的卷积/SDPA 路径同构）：

1. 建描述符：张量 / 算子 / 图；
2. `cudnnBackendFinalize` 提交；
3. 选引擎：用 `CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` 拿候选，再对候选读引擎配置（旋钮 `knobDesc`：tile 大小、Stream-K、kernel cfg …）；
4. `CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR` + `CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR` → `cudnnBackendExecute(handle, plan, variantPack)`。

### 11.2 优势

| 优势 | 含义 | 本机证据 |
| --- | --- | --- |
| 跨算子融合 | 一张图里的多个算子可以由同一个引擎合成少量 kernel，省 HBM 往返与 launch 开销；legacy API 一次调用只能表达一个算子 | SDPA 一次执行覆盖 QK^T / softmax / PV 等多个算子：`logs/fe_attn.log` |
| 全局选引擎 + 可调旋钮 | 引擎与旋钮是一等公民，可按 shape 逐个评估；日志里能看到 `engine_id` 与 `knobDesc`（`KERNEL_CFG` / `TILE_CGA_M` / `STREAM_K` / `TILE_M` / `TILE_N`） | `logs/fe_attn.log`（`engine_id=8` + 旋钮列表） |
| 计划可缓存、可序列化 | 形状与配置不变时复用 `ExecutionPlan`，跳过建图与启发式；图/计划还能导出成 JSON 落盘 | PyTorch `Conv_v8.cpp:285-366` 用线程局部 LRU `BenchmarkCache<cudnn_frontend::ExecutionPlan, ...>` 缓存计划（默认 10000 条，`TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT` 可调）；序列化能力来自 frontend 的 `cudnn_frontend::graph::to_json`（nlohmann::json 符号确实在 `libtorch_cuda.so` 里） |
| 组合式扩展 | 新算子/新融合只需新增描述符与引擎，不必"一个算子加一个 API" | cuDNN 9.x 新增的 attention 等算子仍走同一套 backend API |
| 与布局/设备解耦 | 用 uid + stride + deviceProp 描述，同一张图可在不同 SM 上落到不同引擎 | 引擎自报能力，如 `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`、`..._SUPPORTS_CUDA_GRAPH_NATIVE_API`：`logs/fe_attn.log` |

### 11.3 代价

- 建图本身有开销：demo 里一次卷积前向要 25 次 `cudnnBackendCreateDescriptor`、37 次 `SetAttribute`（§5.4），所以**必须缓存**（PyTorch 靠上面的计划 LRU；demo 只跑一次不需要缓存）。
- API 繁琐、坑多（§5.4 列出 4 个：数组属性不能分两步设置、heuristics 结果要先自建 16 个 engine cfg、从返回的 cfg 读 engine 会 `BAD_PARAM`、`CONVOLUTION_SPATIAL_DIMS` 必需），失败基本只给一个 `CUDNN_STATUS_BAD_PARAM`。
- 形状频繁变化或小算子场景，直接调 legacy API 更划算。PyTorch 因此只在卷积前反向、SDPA 这类"算子大、形状稳定"的路径用 graph API。

### 11.4 只有 graph API 才能选引擎吗？

**不是。legacy API 也能选，只是停在"算法"这一层，而且 cuDNN 9 里它内部仍然走同一套引擎机制。**

| 层次 | 接口 | 能选什么 | PyTorch 怎么用 |
| --- | --- | --- | --- |
| legacy 启发式 | `cudnnGetConvolutionForwardAlgorithm_v7`（BwdData/BwdFilter 各一份） | cuDNN 给出按预估耗时排好序的算法列表，你只能从 `cudnnConvolution*Algo_t` 里挑（fwd 共 8 个：GEMM / FFT / FFT_TILING / IMPLICIT_GEMM / IMPLICIT_PRECOMP_GEMM / DIRECT / WINOGRAD / WINOGRAD_NONFUSED，`cudnn_ops.h:2171` 的 `CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8`） | 默认（`benchmark=False`）只请求 1 个（`Conv_v7.cpp:301-317`），失败再请求全部 8 个兜底 |
| legacy 实测 | `cudnnFindConvolutionForwardAlgorithmEx` | cuDNN 在你的真实张量上把 8 个算法全跑一遍，按实测耗时排序返回 | `torch.backends.cudnn.benchmark=True` 时使用（`Conv_v7.cpp:318-341`），结果进 `BenchmarkCache`（全局 map + mutex，`Conv_v7.cpp:94-130`）缓存 |
| graph / backend API | `CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` + 自建 engine cfg | 枚举全局引擎、读/挑旋钮（tile、Stream-K、kernel cfg）、自己写过滤与计时逻辑；还可用 `CUDNN_ATTR_ENGINE_GLOBAL_INDEX (1301)` 直接钉死某个引擎 | 卷积前反向、SDPA：`Conv_v8.cpp:137 filterEngineConfigs()` 按确定性/TF32 过滤候选，选中计划进线程局部 LRU |

本机证据（`logs/fe_conv_legacy_bench.log`：`TORCH_CUDNN_V8_API_DISABLED=1` + `torch.backends.cudnn.benchmark=True`，脚本 `probe_legacy_bench.py`）：

```
FindConvolutionForwardAlgorithmEx×1、FindConvolutionBackwardDataAlgorithmEx×1、FindConvolutionBackwardFilterAlgorithmEx×1
GetConvolutionForwardWorkspaceSize×8     # 8 个候选算法各查一次显存需求
```

即：benchmark 模式下的 legacy 路径同样在做"选引擎"这件事，只不过候选集被限定为 8 个算法枚举，看不到 engine_id，也调不了旋钮。

两点补充：

1. legacy 内部其实也是同一套引擎机制：`logs/fe_conv_legacy.log:161` 的告警 traceback 里能看到 `outputGraph.buildDAG()` / `initGraph(graph)`，以及 `cudnn::cnn::isForwardSupported(handle, xDesc, wDesc, cDesc, yDesc, algo)` —— cuDNN 在内部把每个 legacy 算法映射到引擎并逐个检查支持性，选中的引擎再由 `cudnnBackendExecuteInternal` 执行（§5.3）。
2. legacy 的选择能力有硬上限：只能选"单个算子的算法"，无法表达跨算子融合图；而 legacy 的选择接口在 cuDNN 9 里已全部标记 `@deprecated ... Use graph API instead`（`cudnn_cnn.h:396`、`cudnn_cnn.h:459`）。

### 11.5 engine 是什么？和硬件有什么关系？

**cuDNN 自己的定义**：`CUDNN_BACKEND_ENGINE_DESCRIPTOR` = "An engine is a **grouping of kernels with similar compute and numerical attributes**"（`cudnn_graph.h:1289`）——即"实现同一张算子图的一种 kernel 分组方案"。**引擎 + 一组旋钮取值 = engine config**（`CUDNN_BACKEND_ENGINECFG_DESCRIPTOR`），engine config + variant pack 才能生成可执行的执行计划。

引擎描述符暴露的属性（`cudnn_graph.h:1286-1332`）：

| 属性 | 含义 |
| --- | --- |
| `CUDNN_ATTR_ENGINE_OPERATION_GRAPH` | 它实现的是哪张图（必填） |
| `CUDNN_ATTR_ENGINE_GLOBAL_INDEX` | 引擎在"全局引擎表"里的编号，取值 0 … `CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT-1`；缓存计划后可以用它钉死同一个引擎 |
| `CUDNN_ATTR_ENGINE_KNOB_INFO` | 性能旋钮（tile 大小、split-K、stage、kernel cfg…），具体取值写在 engine config 的 `KNOB_CHOICES` 里 |
| `CUDNN_ATTR_ENGINE_NUMERICAL_NOTE` | 数值特性：`TENSOR_CORE`（用 Tensor Core）、`DOWN_CONVERT_INPUTS`、`FFT`、`WINOGRAD*`、`NONDETERMINISTIC`、`STRICT_NAN_PROP` …（`cudnn_graph.h:3564-3574`） |
| `CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE` | 行为特性：`RUNTIME_COMPILATION`（需要 NVRTC 现场编译）、`REQUIRES_FILTER_INT8x32_REORDER`、`CUBLASLT_DEPENDENCY`、`SUPPORTS_CUDA_GRAPH_NATIVE_API`（`cudnn_graph.h:3581-3586`） |
| `CUDNN_ATTR_ENGINE_LAYOUT_INFO` | 它偏好的张量布局 |
| `CUDNN_ATTR_ENGINE_SM_COUNT_TARGET` | 目标 SM 数量（0 = 全部 SM） |
| `CUDNN_ATTR_ENGINE_DEVICEPROP` | **这条引擎针对哪块设备** |

**和硬件有关吗？强相关——但 engine 本身是软件，不是硬件单元。** 三层证据：

1. **引擎记录里直接写着 SM 版本**：本机所有引擎日志都是 `"smVersion":1200`（RTX 5060 = sm_120，`graph_conv/graph_conv.log`）：
```
"engine":{"engineId":28,"smVersion":1200,"knobChoices":{"CUDNN_KNOB_TYPE_TILE_SIZE":4,"CUDNN_KNOB_TYPE_USE_TEX":0}}
```
2. **引擎被按 GPU 架构预编译或现场 JIT**（`nvidia/cudnn/lib/`）：
   - `libcudnn_engines_precompiled.so.9`（248 MiB）：二进制里嵌着 `sm_75 / sm_80 / sm_86 / sm_90a / sm_100f / sm_120f / sm_121` 等按架构切分的 kernel；
   - `libcudnn_engines_tensor_ir.so.9`（192 MiB）：导入 `__nvrtcNvvmHandle`，按架构族（`sm_8x / sm_9x / sm_10x`）生成；
   - `libcudnn_engines_runtime_compiled.so.9`（32 MiB）：导入 `nvrtcHelper::load` / `RuntimeKernel::getNvrtcVersion`，对应 `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`——本机 SDPA 选中的引擎就带这个 flag（`logs/fe_attn.log`）；
   - `libcudnn_heuristic.so.9`（58 MiB）：引擎表的启发式筛选。
3. **可用引擎数随设备变**：`CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT`（`cudnn_graph.h:800,1847`）是"在**当前设备上**支持这张图的引擎数量"，换 GPU 或换 cuDNN 版本，编号就可能变——所以全局编号只能在同一设备+同一 cuDNN 版本内复用。

**同一张图在本机的引擎编号是稳定的**（Conv2d 4x3x64x64 → 4x16x64x64，pad=1，三个独立进程；编号的完整语义——它是什么、随什么变、能拿来干什么——见 §11.11）：

| 算子 | engine_id |
| --- | --- |
| forward | 28 |
| backward data | 0 |
| backward filter | 50 |

v8 frontend 路径（`logs/fe_conv.log`）与 legacy 路径内部（`logs/fe_conv_legacy.log`）落到的是**同一批引擎**；benchmark 模式（`logs/fe_conv_legacy_bench.log`）还会出现 3/4/5/6/7/17/18/20 等编号，那是 `cudnnFindConvolution*AlgorithmEx` 把候选引擎真跑了一遍留下的痕迹。demo 的启发式候选里则出现 0/1/4/28/35/36/38/46/48（`graph_conv/graph_conv.log`）。

**别混淆**：NVIDIA 生态里的 "engine" 还有另一个含义——GPU 上的**硬件单元**（copy engine / compute engine / NVENC / DLA，CUPTI 的 "engine" 计数器也是指这些）。cuDNN 的 engine 是"kernel 分组 + 旋钮 + 数值/布局行为"的**软件方案**：它**使用**硬件特性（Tensor Core、SM 数量、SM 版本、TMA 等），但本身不是一块硅。

> 实测普查：本机一张 4x3x64x64→4x16x64x64 的卷积图有 81 / 85 / 75 个 engine（fwd/bwd_data/bwd_filter），
> 其中只有 25/33/31 个能在 sm_120 上 finalize；heuristic 只排 17/12/16 个候选，top-1 正是 PyTorch 实跑用到的
> engine 28 / 0 / 50。逐个 engine 的 kernel、精度、耗时对照见 §11.9。

### 11.6 别把三种 "graph" 搞混

| 名称 | 所在层 | 解决什么问题 |
| --- | --- | --- |
| cuDNN graph / backend descriptor API | cuDNN | 用算子 DAG 描述计算，由 cuDNN 选引擎并做融合 |
| CUDA Graph | CUDA runtime | 把一串已确定的 kernel launch 一次性提交，省的是 launch 开销，不改变算子本身 |
| torch.compile 的 FX graph | PyTorch 前端 | Python 层捕获计算图并交给后端编译，最终可能落到 cuDNN graph 或别的 kernel |

> 三者可以叠加：cuDNN 引擎会自报是否支持 CUDA Graph 原生 API（日志里的 `CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API: true`，`logs/fe_attn.log`）。

### 11.7 执行计划（plan）阶段到底做了什么

一句话：**plan 阶段把"引擎 + 旋钮"落实成"这台设备上可执行的 kernel + 启动配置 + workspace 需求"**，而且这一步的结果可以缓存、可以序列化。

| plan 阶段做的事 | 依据 |
| --- | --- |
| 绑定 handle 与 engine config（都是必填属性），校验该引擎在当前设备/这张图上可用 | `CUDNN_ATTR_EXECUTION_PLAN_HANDLE`（`cudnn_graph.h:1418`）、`..._ENGINE_CONFIG`（`:1421`） |
| **落实 kernel 实现**：预编译引擎 = 加载对应的 SASS；runtime-fusion 引擎 = NVRTC 现场编译 | 引擎的 `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`；耗时差异见下表 |
| 算出 workspace 需求（只读属性，finalize 之后才能读） | `CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE`（`:1437`） |
| 提供 kernel cache 复用点：相同融合图可跳过重复编译 | `CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE`（`:1447`，cuDNN 9.4+） |
| 整个计划可序列化成 JSON、也可反序列化回来 | `CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION`（`:1441`） |
| **不做**的事：绑数据指针（执行时由 variant pack 绑，`:1884`）、选 stream（由 handle 带） | `cudnnBackendExecute(handle, plan, variantPack)` |

实测：同一台机器上，plan finalize 的耗时几乎完全由"这个引擎要不要现场编译"决定。

| 场景 | engine_id | runtime 编译 | plan finalize 耗时 |
| --- | --- | --- | --- |
| graph demo（1x3x8x8 卷积，`graph_conv/graph_conv.log`） | 0 | 否 | **0.021 ms** |
| PyTorch Conv2d 前向 / dgrad / wgrad（`logs/fe_conv.log`） | 28 / 0 / 50 | 否 | 0.011 / 0.009 / 0.010 ms |
| PyTorch SDPA（`logs/fe_attn.log`） | 8 | **是** | **51.5 ms** |

SDPA 那次进程日志总跨度 127.7 ms，其中一条 plan finalize 就占 51.5 ms（约 40%）——这就是"plan 很贵、必须缓存"的来源，也解释了 PyTorch 为什么把选中的 `cudnn_frontend::ExecutionPlan` 放进线程局部 LRU（§11.2）。

三个阶段的分工：

| 阶段 | 输入 | 产物 |
| --- | --- | --- |
| 选引擎 | 算子图 + heuristics | 候选 engine config（含旋钮） |
| **plan** | handle + engine config | 可执行的 kernel + 启动配置 + workspace 大小（可缓存/序列化） |
| execute | handle + plan + variant pack | 真正下发的 kernel launch（日志里落到 `cudnnBackendExecuteInternal`），细节见 §11.8 |

### 11.8 execute 阶段到底做了什么

一句话：**execute 只是"把 plan 里已经编译/加载好的 kernel，用 variant pack 里的指针，下发到 handle 绑定的 stream 上"**——它返回不代表算完。

| 做的事 | 依据 |
| --- | --- |
| 收三件输入：handle（带 stream）、plan（kernel + 启动配置）、variant pack（uid → 设备指针、workspace） | execute 记录的三个段落 `handle:` / `executionPlan:` / `variantPack:` |
| 把 uid 映射成设备指针并校验（uid 是否齐全、指针对齐、workspace 够不够） | `CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS`（`cudnn_graph.h:1884`） |
| 在 **handle 的 stream** 上发射 kernel（异步），随即返回 | 日志 footer 里的 `StreamId=0x...`；耗时见下 |
| 每次 execute 落地一条 `cudnnBackendExecuteInternal`，里面是解析后的执行形态（`operation=CONV_FORWARD`、`engine_id`、旋钮取值、`alpha/beta`、`target_sm_count`） | `graph_conv/execute_async.log` |
| **不做**：不选引擎、不编译 kernel、不分配 workspace、不做任何同步 | — |

实测（`graph_conv/execute_async_demo.cpp`：1x128x1024x1024 * 128x128x3x3，padding=1，engine_id=38）：

```
第 1 次：execute 返回 297 us，之后 synchronize 又等了 24761 us
后 3 次：execute 返回平均 340 us，GPU 上真正跑完还要 25314 us
```

host 侧 ~0.3 ms 就把活交出去了，GPU 还要跑 ~25 ms —— 这就是"execute 是异步下发"的直接量化。

variant pack 在日志里的样子（uid → 设备地址）：

```
variantPack: type=CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
    Uid=0: location=dev; addr=0x506200000;      # x
    Uid=1: location=dev; addr=0x505e01600;      # w
    Uid=2: location=dev; addr=0x526200000;      # y
    Workspace: location=dev; addr=0x546200000;
```

顺带一个容易踩的点：workspace 是 plan 报出来的（这个例子 **537 MB**），必须在 execute 之前由调用者准备好——PyTorch 是用它的 caching allocator 出这块显存。

### 11.9 engine 都干了什么、彼此有什么区别（本机普查 + 真跑对照）

先给结论：**同一张图上的所有 engine 算的是同一个卷积，区别只在于"用什么 kernel、配什么旋钮、摆成什么布局、算到多少精度、要多少 workspace"**。engine 描述符里的四个只读标签（numerical / behavior / layout / SM target）就是 cuDNN 给这些差别打的"自述标签"，而 `heuristic` 负责按"跑得快"给它们排序、只把少数几个交给你。

#### 11.9.1 全局 engine 表：数量、可用性、标签分布

建图后先读 `CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT`，再对 `i = 0 … count-1` 逐个 `CUDNN_BACKEND_ENGINE_DESCRIPTOR` + `CUDNN_ATTR_ENGINE_GLOBAL_INDEX = i` finalize，即可拿到每个 engine 的标签（工具：`engine_enum/engine_enum.cpp`，输出：`engine_enum/engine_enum_{fwd,bwd_data,bwd_filter}.txt`）：

| 卷积图（N=4,C=3,64x64 → K=16,R=S=3,pad=1） | 全局 engine 数 | 本机能 finalize（可用） | 失败类型 | TensorCore | FFT | Winograd | RUNTIME_COMPILATION | 非确定性 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 81 | 25 | ARCH_MISMATCH 42 / NOT_SUPPORTED 14 | 16 | 2 | 1 | 5 | 0 |
| backward data | 85 | 33 | 41 / 11 | 22 | 3 | 1 | 5 | 1 |
| backward filter | 75 | 31 | 34 / 10 | 24 | 1 | 0 | 5 | 1 |

- "全局 engine 数"包含**为其它 GPU 架构预编译**的 kernel（`ARCH_MISMATCH`）和当前版本不支持的组合（`NOT_SUPPORTED`）——本机 sm_120 上真正能用的只有 1/3 左右。
- 标签是**互不排斥的集合**：一个 engine 可以同时是 `TENSOR_CORE + DOWN_CONVERT_INPUTS`（= 用 Tensor Core 算，但把 FP32 输入降精度成 TF32）或 `TENSOR_CORE + RUNTIME_COMPILATION`（JIT 出来的 Tensor Core kernel）。
- `RUNTIME_COMPILATION` 的那 5 个（本机 fwd 里是 5/71/72/79/80）就是 `libcudnn_engines_runtime_compiled.so.9` 的 NVRTC 引擎，它们**没有**进 heuristic 的候选表——小图用不着现场编译。§5.6 里 SDPA 命中的引擎（`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16…`）就是这一类。

fwd 图里 25 个可用 engine 的标签 + engine config 报出的 workspace（同一张图的**差异一览**）：

| engine | numerical_note | behavior_note | workspace | 说明 |
| --- | --- | --- | --- | --- |
| 0 | STRICT_NAN_PROP | — | 0 | 直接法，NaN 严格传播 |
| 1 / 28 | — | — | 108 / 0 | 经典 `implicit_convolve_sgemm`（FP32），PyTorch 常选 |
| 2 | — | — | 1769472 | 需要 workspace 的老式实现 |
| 3 / 4 | FFT | — | 8521408 / 870400 | 频域卷积 |
| 5 | — | RUNTIME_COMPILATION | -（要旋钮） | NVRTC 现场编译 |
| 6 | WINOGRAD | — | 11213568 | Winograd 4x4 |
| 22 / 23 / 27 / 45 / 47 | TENSOR_CORE,DOWN_CONVERT_INPUTS | — | 788752 … 623472 | TF32 implicit GEMM |
| 35 / 36 / 38 / 39 / 41 / 46 / 48 / 58 | TENSOR_CORE | — | 1313040 / 264464 / 1246928 … | TF32/xmma/cutlass |
| 71 / 72 / 79 / 80 | TENSOR_CORE（80 无） | RUNTIME_COMPILATION,SUPPORTS_CUDA_GRAPH_NATIVE_API | 1247296 … | JIT 引擎 |

> 注意：`workspace` 是**不带旋钮**的 engine config 报出来的下限；带旋钮的候选值见 §11.9.3。

#### 11.9.2 heuristic 只交给你一小撮，而且 top-1 就是你实际用到的那条

heuristic（`CUDNN_HEUR_MODE_A`）返回的是**排好序的候选 engine config**，数量远小于全局表：

| 图 | 全局 engine | heuristic 候选数 | top-1 engine | PyTorch 实跑的 engine_id（§4.7.2 实测） |
| --- | --- | --- | --- | --- |
| forward | 81 | 17 | **28** | 28 ✓ |
| backward data | 85 | 12 | **0** | 0 ✓ |
| backward filter | 75 | 16 | **50** | 50 ✓ |

**候选的 engineId 和旋钮要从 execution plan 的 JSON 里读**（`CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION`）：

```json
"engine":{"engineId":28,"smVersion":1200,
          "knobChoices":{"CUDNN_KNOB_TYPE_TILE_SIZE":4,"CUDNN_KNOB_TYPE_USE_TEX":0}}
```

一份候选表（fwd，节选；完整见 `engine_enum/engine_enum_fwd.txt`）：

| # | engineId | 旋钮 | workspace |
| --- | --- | --- | --- |
| 0 | 28 | TILE_SIZE=4, USE_TEX=0 | 0 |
| 1 | 28 | TILE_SIZE=3, USE_TEX=0 | 0 |
| 2 | 48 | TILE_SIZE=2, TILEK=2, STAGES=0, LDGA/LDGC | 1246928 |
| 4 | 1 | TILE_SIZE=2, USE_TEX=0 | 108 |
| 5 | 6 | WORKSPACE=0 | 11213568 |
| 8 | 0 | （无旋钮） | 0 |
| 11 | 36 | TILE_SIZE=0, TILEK=2, STAGES=3, IDX_MODE=1, SPECFILT=0 | 1313040 |
| 12 | 38 | TILE_SIZE=0, TILEK=2, STAGES=3, IDX_MODE=0, LDGC=0 | 264464 |
| 15 | 3 | MULTIPLY=0 | 8521408 |
| 16 | 4 | （无旋钮） | 870400 |

bwd_filter 的候选旋钮里还会出现 `SPLIT_K`（split-K 并行度，取值 4/24/35/150），bwd_data 里出现 `STREAM_K` 类旋钮——**旋钮空间本身就是 engine 的一部分**（`CUDNN_ATTR_ENGINE_KNOB_INFO` 只给它允许的类型/上下界，取值在 engine config 里）。

#### 11.9.3 把候选真跑一遍：kernel / 精度 / 耗时对照

工具：`engine_enum/runner.cpp`（编译成 `librunner.so`，用 ctypes 在进程内执行某个候选）+ `engine_enum/trace_engines.py`（`torch.profiler(ProfilerActivity.CUDA)` 抓 kernel，并把 kernel 名在 cuDNN 各子库里做字符串搜索标注归属）。20 次迭代平均，同图同数据，和 CPU 参考卷积比精度：

| 候选 | engineId | 主 kernel（节选） | 发射次数/迭代 | ms/迭代 | maxerr | 归属库 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 28 | `implicit_convolve_sgemm<float,float,int,1024,5,5,3,3,3,1,…>` | 1 | 0.019 | 2.4e-06 | engines_precompiled |
| 1 | 28 | `implicit_convolve_sgemm<…,128,5,5,3,3,3,1,…>`（TILE_SIZE=3） | 1 | 0.024 | 2.4e-06 | engines_precompiled |
| 7 | 28 | `implicit_convolve_sgemm<…,512,6,8,3,3,5,1,…>`（TILE_SIZE=1） | 1 | 0.31 | 2.4e-06 | engines_precompiled |
| 2 | 48 | `nchwToNhwc` ×2 → `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32…` → `nhwcToNchw` | 4 | 0.038 | 3.3e-03 | engines_precompiled |
| 4 | 1 | `kern_precompute_indices` + `precomputed_convolve_sgemm<…,512,6,7,4,…>` ×4 | 6 | 0.33 | 2.4e-06 | engines_precompiled |
| 5 | 6 | `winogradForwardData4x4` / `winogradForwardFilter4x4` / `gemmSN_NN_kernel` / `winogradForwardOutput4x4` | 4 | 0.13 | 1.4e-05 | engines_precompiled |
| 6 | 46 | `nchwToNhwc` → `cutlass__5x_cudnn::Kernel<…s1688fprop_optimized_tf32_256x64_16x4…>` → `nhwcToNchw` | 4 | 0.060 | 3.3e-03 | engines_precompiled |
| 8 | 0 | `cudnn::cnn::conv2d_grouped_direct_kernel<…>` | 1 | 0.057 | 2.4e-06 | engines_precompiled |
| 11 | 36 | `nchwToNhwc` → `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32…tilesize128x128x16_stage…` → `nhwcToNchw` | 4 | 0.074 | 3.3e-03 | engines_precompiled |
| 12 | 38 | `nchwToNhwc`×2 → `sm80_xmma_fprop_implicit_gemm_tf32f32…nhwckrsc_nchw…`（输出直出 NCHW，省掉转回） | 3 | 0.10 | 3.3e-03 | engines_precompiled |
| 14 | 35 | `nchwToNhwc` → `cutlass__5x_cudnn::Kernel<…s1688fprop_analytic_tf32_128x64_32x3…>` → `nhwcToNchw` | 4 | 0.087 | 3.3e-03 | engines_precompiled |
| 15 | 3 | `DSE::regular_fft_pad` / `DSE::vector_fft` / `region_transform_ABC_val` / `cutlass_100_simt_cgemm`（FFT） | 10 | 1.16 | 5.2e-06 | engines_precompiled |
| 16 | 4 | `fft2d_r2c_32x32` / `cutlass_100_simt_cgemm` / `fft2d_c2r_32x32`（FFT） | 28 | 0.42 | 4.8e-06 | engines_precompiled |

读法（这就是"engine 之间到底差在哪"的实证）：

- **换算法**：直接法（engine 0）/ 隐式 GEMM（1、28）/ Winograd（6）/ FFT（3、4）/ cutlass-xmma TensorCore（35/36/38/46/48）。
- **换旋钮**：同样是 engine 28，`TILE_SIZE` 1/3/4 对应模板参数 `512,6,8` / `128,5,5` / `1024,5,5` 三种 tile——精度完全一样（2.4e-06）。本表是**单次 20 迭代**采样（0.31 / 0.024 / 0.019 ms），受本机时钟抖动影响大；稳定复测见 §11.12.4（200 迭代 ×3 取最好：0.052 / 0.015 / 0.018 ms，差最多 ~3.5 倍）。
- **换布局**：TensorCore 引擎要求 NHWC，于是每迭代多出 `nchwToNhwc` + `nhwcToNchw`（可省略时——引擎 38——就少一次转换，3 次发射）。
- **换精度**：TF32 引擎的 maxerr ≈ 3.3e-03，FP32 引擎 ≈ 2.4e-06，差三个数量级；Winograd 1.4e-05；FFT 5e-06。
- **换 workspace**：0 字节（engine 0/28）到 11 MiB（Winograd）。
- **换发射形态**：1 次 launch（sgemm/direct）到 588 次 launch（FFT 引擎 4，20 次迭代 = 每次迭代 28~29 个 kernel）。

> 复跑命令：`engine_enum/run_engine_enum.sh`（编译 + 三个图的普查 + 17 个候选抓 kernel），原始输出在 `engine_enum/{engine_enum_*.txt,trace_all.txt,kernels_cand*.txt,engines_kernels.txt}`。

#### 11.9.4 谁提供这些 engine：三个引擎库 + heuristic

- 本次 17 个候选**发的每一个 kernel 都能在 `libcudnn_engines_precompiled.so.9` 里搜到**（sgemm / direct / winograd / fft / xmma / cutlass 模板都在里面）——即当前这些 engine 都是"预编译引擎"。
- `libcudnn_engines_tensor_ir.so.9` 的引擎以 Tensor IR 描述、按架构族现场生成，`libcudnn_engines_runtime_compiled.so.9` 的引擎整段 NVRTC 现场编译（kernel 名里带 `cudnn_generated_*`，见 `launch_probe/kernels_attention_forced.txt`）。它们同样注册进全局 engine 表（带 `RUNTIME_COMPILATION` 标签），只是这张小图没被选中。
- `libcudnn_heuristic.so.9` 只做排序/筛选（§4.7.1），它把候选 config 交回 graph，由 graph 组装 plan；真正发射 kernel 的是引擎库自己（§4.7.2：引擎库 dlsym `cuLaunchKernel`）。

#### 11.9.5 复现时的三个坑（都踩过）

1. 只读属性的 count 查询口径不一致：多数属性要 `data=nullptr`，而 `CUDNN_ATTR_ENGINEHEUR_RESULTS` 需要**非空**指针才能拿到个数，且取回时**必须自己先创建好 descriptor 数组**（否则 cuDNN 往你的缓冲区写指针 → 踩内存/段错误）。
2. heuristic 返回的 engine config 是**借用句柄**，不能再对它 `GetAttribute`（`ENGINE` 报 `BAD_PARAM`，`KNOB_CHOICES` 报 0 甚至崩溃）；要 engineId/旋钮就走 execution plan 的 JSON。
3. 读 `CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE` 时不要把 count 和 data 用同一个变量接（会拿到 1），否则 workspace 分配过小、`cudnnBackendExecute` 报 `CUDNN_STATUS_EXECUTION_FAILED_CUDART`。

### 11.10 `libcudnn_engines_precompiled` 与 `libcudnn_engines_runtime_compiled` 的分工

先给一句话：**两个库都是"engine 供应商"，把各自的 engine 注册进同一张全局 engine 表**（§11.9.1 里那张表就是两家合起来的）；区别在于 **kernel 是"出厂就编译好的"还是"用的时候现场生成的"**。

| | `libcudnn_engines_precompiled.so.9` | `libcudnn_engines_runtime_compiled.so.9` |
| --- | --- | --- |
| 文件大小 | **248 MiB** | **32 MiB** |
| 里面装的东西 | 各架构的现成 cubin（SASS）+ 选核逻辑 | **CUDA 源码模板**（带 `{占位符}`）+ 代码生成器 + 编译/缓存驱动 |
| 标志性字符串 | `implicit_convolve_sgemm`×400、`winograd`×77、`fft2d_r2c`×18、`sm80_xmma_fprop`×978、`s1688fprop`×45 | `cudnn_generated`、`fort_native`、`CUTLASS_NVRTC_ENABLE`、`useNvrtcSassPath`×12、`FORT`×379 |
| 导出符号 | 2 个（`dummyBackendEnginesPreCompiledSymbol` + soname） | 11 个，含 **`registerRuntimeCompiled`** 和 `cudnn::fusion::{setCustomLutPath, calibrateCustomLut, getRtcLutKeyString, saveCustomLutEntry, invalidateRtcCustomLutCache…}`（编译结果的 LUT/缓存管理 API） |
| 编译器依赖 | 无 | 从 `libcudnn_graph` 导入 `nvrtcHelper::load`、`RuntimeKernel::getNvrtcVersion`；`libcudnn_graph` 里就是 NVRTC 的加载器（`"libnvrtc.so.%d"` / `"Could not open libnvrtc for runtime linking…"`）和开关 `CUDNN_NVRTC_RUNTIME_LINKING` |
| 引擎标签 | 普通 engine（可能带 `TENSOR_CORE`/`FFT`/`WINOGRAD`/`DOWN_CONVERT_INPUTS`） | 带 **`CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`** 的 engine（本机 conv 图上是 5/71/72/79/80） |
| kernel 名 | 模板原样，如 `implicit_convolve_sgemm<float,float,int,1024,5,5,3,3,3,1,…>` | 现场拼出来，如 `cudnn_generated_fort_native_sm80_convFwd_knob_21_128x128x16_4_0`、`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_1_128x64…` |

**动态对照（同一张卷积图 4x3x64x64→4x16x64x64，同一数据，`engine_enum/trace_engine_global.py` 实测）**：

| engine | 归属 | plan finalize（含选核/编译） | 执行 | 主 kernel |
| --- | --- | --- | --- | --- |
| 0 | precompiled | **0.02 ms** | 0.04 ms | `cudnn::cnn::conv2d_grouped_direct_kernel<…>` |
| 28 | precompiled | **0.02 ms** | 0.03 ms | `implicit_convolve_sgemm<…>` |
| 1 / 2 | precompiled | 0.05 / 0.07 ms | 0.12 / 0.04 ms | `implicit_convolve_sgemm` + `precomputed_convolve_sgemm` |
| 6 | precompiled | 0.13 ms | 0.09 ms | `cudnn::winograd_nonfused::winogradForward*4x4` |
| 48 | precompiled | 0.06 ms | 0.04 ms | `sm80_xmma_fprop_implicit_gemm…tf32…`（+布局转换） |
| **71** | **runtime_compiled** | **19.9 – 39.7 ms**（首次冷启动 478 ms） | 0.07 ms | `cudnn_generated_fort_native_sm80_convFwd_knob_21_128x128x16_4_0` |

- 代价差 **3 个数量级**：预编译引擎每建一个 plan 只要 **0.02–0.2 ms**（换个没见过的 batch=9 也一样），runtime 引擎要 **20–40 ms**——这笔钱花在 **plan finalize**，不在 execute（execute 还是 ~0.1 ms 级别的下发）。之前 SDPA 那次实测的 51.5 ms 也是同一笔钱（§5.6）。
- 换一个全新的 shape（N=4 → N=8）不会让预编译引擎变慢，但 runtime 引擎仍然每次 plan 都付 20–40 ms；同一个 shape 重复建 plan 也一样（代价按 plan 付，不按进程付）。
- 证据链：`engine 71` 的 kernel 名带 `cudnn_generated_fort_…`，而 `libcudnn_engines_runtime_compiled.so.9` 里存的正是带 `{umma_arrive_fn}`、`{bytes_per_ld_k}`、`Tile_K` 这类占位符的 **CUDA 源码模板**（`strings` 可见），只有它带 `CUTLASS_NVRTC_ENABLE` / `useNvrtcSassPath` 这些 NVRTC 编译开关。

**各自到底负责什么**

- `precompiled` = **主力**：卷积前/反向、Winograd、FFT、TF32/xmma/CUTLASS 这些"高频且有谱"的实现全都预编译好，覆盖各代架构；heuristic 排出来的候选绝大多数来自它（§11.9.3 里 17 个候选的 kernel 全部能在它里面搜到）。它自己不做代码生成，只做"挑一个现成 cubin + 配置 grid/block/smem"。
- `runtime_compiled` = **补位 + 融合**：当某个 shape/精度/融合组合没有划算的预编译实现时才上场，典型是 flash-attention 系（SDPA）和 FORT（Fusion Runtime）融合算子；它把源码模板拼成 CUDA 源码 → 编译（NVRTC / 驱动 JIT）→ 缓存（`~/.nv/ComputeCache` 在 runtime 引擎跑过之后有新条目；库本身还导出 `setCustomLutPath/calibrateCustomLut/…` 这套编译结果 LUT 管理 API）。**代价是一次性的编译时间，收益是能用上针对你这张图现生成的 kernel。**
- 顺带把第三个库也放进来对照：`libcudnn_engines_tensor_ir.so.9`（192 MiB）导入 `__nvrtcNvvmHandle`，内部还带 ptxas 的痕迹——它走的是"Tensor IR → 生成代码"的中间路线，是 cuDNN 10/未来融合算子的主力形态。
- 三个库在进程里都是**被 graph 库 dlopen 进来的**（`graph_conv/ld_libs.log`：一个纯卷积程序也会把 precompiled / runtime_compiled / tensor_ir / heuristic 全部加载），所以"哪个库被加载了"不能说明引擎来自谁——要看引擎的 `BEHAVIOR_NOTE` 和实际 kernel 名字。

**未完全定位的一点**：SDPA 那条路跑完，进程里能直接看到 `libnvrtc.so.13` + `libnvrtc-builtins.so.13.0` 的映射；但 conv 图的 engine 71 跑完看不到 nvrtc 映射（只看到 `~/.nv/ComputeCache` 被写入），推测是"生成 PTX → 驱动 JIT + 缓存"或"NVRTC 用完即 dlclose"路径，留待后续确认（不影响上面的结论：它的 kernel 一定是现场生成的，plan 时间也确实是 20–40 ms）。

### 11.11 engine 的编号（`CUDNN_ATTR_ENGINE_GLOBAL_INDEX`）是什么意思

**一句话**：编号是 **"这张算子图的引擎表里的下标"**——`0 … CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT-1` 的稠密区间，由 cuDNN 在建图时分配；它唯一标识"在**这份图 + 这个设备 + 这个 cuDNN 版本**下的一条引擎实现"，出了这个上下文就不保证还是同一个东西。

**它不是什么**

| 常见误解 | 事实 |
| --- | --- |
| "编号 = 性能排名 / 优先级" | 不是。排名是 heuristic 结果数组的**顺序**（§11.9.2 的 `#0…#16`），同一个 engine 可以凭不同旋钮占多个名次（候选 0/1/7 全是 engine 28） |
| "编号 = 算法（legacy `cudnnConvolutionFwdAlgo_t`）" | 不是。legacy 的 8 个 algo 是**另一套**枚举，cuDNN 内部把它们映射到引擎（`legacy_conv_bench` 里 `Find...Ex` 会把 0/3/4/5/6/7/17/18/20/28/50 这些引擎挨个试） |
| "编号 = 一个 kernel" | 不是。一个 engine 是**一组 kernel + 启动配置**：engine 50（wgrad）每迭代发 5 个 kernel，engine 28 换 `TILE_SIZE=1/3/4` 会换成完全不同的模板参数，编号仍是 28 |
| "同一个编号在不同图里是同一个引擎" | **不是**。实测 #28：在 fwd 是 FP32 `implicit_convolve_sgemm`（无标签、workspace=0），在 bwd_data 是 `TENSOR_CORE`（workspace=1313040）。三个图共有的 75 个编号里，只有 23 个的标签完全一致 |

**范围和稳定性（实测）**

| 问题 | 实测结论 |
| --- | --- |
| 编号区间 | 恒为 `0..count-1` 稠密（每个下标都能建出 engine 描述符，要么 finalize 成功，要么报 `ARCH_MISMATCH`/`NOT_SUPPORTED`） |
| count 由什么决定 | **图的结构**（算子种类/数据类型/布局/设备/版本），**与形状无关**：N=4/8/9/16、C=3/64、64x64 与 224x224、K=3/16/64 全是 **81**；fwd / bwd_data / bwd_filter 分别是 **81 / 85 / 75** |
| 跨进程稳定吗 | 稳定。两个相隔 40 分钟、二进制版本不同的进程，81 行逐列一致；§11.5 里 28/0/50 也是三次独立进程一致 |
| 跨版本/设备稳定吗 | 不保证。所以 plan 的 JSON 必须连 `cudnnVersion:92400`、`smVersion:1200`、`engineId`、`knobChoices` 一起存——少了版本号，这个编号就没有意义 |
| 表里都是能用的吗 | 不是。本机 fwd 81 个里只有 25 个能 finalize，其余是给别的架构预编译（`ARCH_MISMATCH`）或本版本不支持（`NOT_SUPPORTED`） |

**它能拿来干什么**

1. **钉死一条路径**：建 `CUDNN_BACKEND_ENGINE_DESCRIPTOR` + `CUDNN_ATTR_ENGINE_GLOBAL_INDEX = i` 就能绕开 heuristic 直接跑第 i 号引擎（本仓库的 `engine_enum/engine_enum.cpp`、`runner.cpp::run_engine_global` 就是这么做的）——复现性能问题、对比 kernel 时非常有用。
2. **缓存/序列化**：日志与 execution plan JSON 里都带它（`"engine":{"engineId":28,…}`），`cudnn_frontend` 的 `EngineConfig` 也是靠它做 key。
3. **读日志**：`cudnnBackendExecuteInternal` 记录里的 `engine_id: type=int; val=28;` 就是这次执行用的引擎编号；PyTorch 报的 "cudnn engine id" 也是它。

**怎么自己读一遍**（命令见 `engine_enum/README.txt`）：

```
CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT  -> 81
for i in 0..80:  engine 描述符 + GLOBAL_INDEX=i  ->  numerical/behavior note、workspace、能否 finalize
heuristic 候选 + execution plan JSON              ->  engineId=28 + 旋钮 {TILE_SIZE:4, USE_TEX:0}
```

### 11.12 engine 与旋钮（knob）的关系：旋钮不是"调参旋钮"，是"模板实例选择器"

#### 11.12.1 概念：engine 是算法/模板家族，旋钮是模板实参

- **engine**（`CUDNN_BACKEND_ENGINE_DESCRIPTOR` 的全局编号）= 一组可选的 kernel 实现（§11.9）。
- **旋钮（knob）**= 从这组实现里挑**哪一个具体实例**的参数：tile 大小、流水级数、split-K、精度/布局档、workspace 档。
- 三件套（`cudnn_graph.h` 的 716 / 737-738 / 835-849 / 1302 / 1524 行）：
  - `CUDNN_ATTR_ENGINE_KNOB_INFO`（engine 的**只读**属性，返回 0..N 个 `CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR`）：每个旋钮的 `TYPE` + `[MINIMUM..MAXIMUM]` + `STRIDE`；
  - `CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR` = 一对 `(CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE)`；
  - `CUDNN_ATTR_ENGINECFG_KNOB_CHOICES`：把若干这样的"取值对"塞进 engine config。
- 于是 **engine + 一组旋钮取值 = engine config**，execution plan 只能由 engine config 建；`engineId` 与 `knobChoices` 只在 plan 的 JSON 里能看到（heuristic 返回的 config 是借用句柄，不能再 `GetAttribute`，§11.9.5）。

最直观的实证（`engine_enum/knob_sweep.py` + `knob_kernels.txt`）：同一个 engine 28，只改 `TILE_SIZE`，cuDNN 发出的就是同一个函数模板的不同实例：

| engine 28 的旋钮 | CUPTI 抓到的 kernel（模板实参随之变） | ms/iter（200×3 取最好） |
| --- | --- | --- |
| TILE_SIZE=0 | `implicit_convolve_sgemm<float,float,int,128,6,7,3,3,5,1,false,false,true>` | 0.0302 |
| TILE_SIZE=1 | `implicit_convolve_sgemm<…,512,6,8,3,3,5,1,false,false,true>` | 0.0524 |
| TILE_SIZE=2 | `implicit_convolve_sgemm<…,1024,6,7,3,3,5,1,false,false,true>` | 0.0292 |
| TILE_SIZE=3 | `implicit_convolve_sgemm<…,128,5,5,3,3,3,1,false,false,true>` | 0.0149 |
| TILE_SIZE=4 | `implicit_convolve_sgemm<…,1024,5,5,3,3,3,1,false,false,true>` | 0.0178 |
| TILE_SIZE=4 + USE_TEX=1 | 同模板，第 11 个 bool 实参翻成 `true` | 0.0389 |

六种取值精度完全一样（maxerr=2.38e-06，都是 FP32 直接卷积）——**旋钮只换"用哪份预编译实例"**。计时见 `knob_bench.txt`（同一会话内可比；绝对值和时钟一起漂，见 §11.12.4）。

#### 11.12.2 每个 engine 的旋钮名单（本机普查：`knob_info_{fwd,bwd_data,bwd_filter}.txt`）

工具 `engine_enum/knob_info.cpp`（用法 `./knob_info fwd|bwd_data|bwd_filter [起始] [结束]`）把三个图里每个能 finalize 的 engine 的旋钮全列了出来。fwd 图 25 个可用 engine 的旋钮签名去重后只有 11 种：

| 旋钮签名（`[min..max]`） | fwd 图上的 engine |
| --- | --- |
| （无旋钮） | 0、4、5 |
| TILE_SIZE[0..4]、USE_TEX[0..1] | 1、2、28 |
| MULTIPLY[0..2] | 3 |
| WORKSPACE[0..2] | 6、58 |
| TILE_SIZE[0..23]、TILEK[0..3]、STAGES[0..7]、IDX_MODE[0..1]、SPECFILT[0..4] | 22、36 |
| 上行 + LDGC[0..4] | 38 |
| 再加上 SPLIT_K_BUF[1..2048]、SPLIT_K_SLC[1..2048]、REDUCTION_MODE[0..1] | 23、39（41 = 再叠 LDGC） |
| TILE_SIZE、KBLOCK[0..5]、STAGES | 27、35、45、46 |
| TILE_SIZE、LDGA[0..4]、TILEK、STAGES、LDGC | 47、48 |
| SPLIT_K_SLC[1..128]、KERNEL_CFG[0..N]（runtime 编译组；71/72/79 另有 STREAM_K[1..2]） | 71(N=41)、72(N=55)、79(N=263)、80(N=0，无 STREAM_K) |

- **旋钮同名不同义**：`TILE_SIZE` 在 engine 28 上是 `[0..4]`，在 engine 46 上是 `[0..23]`（`TILEK`/`STAGES` 一起用），在 bwd_filter 的 engine 1 上是 `[0..8]`——**跨 engine 套数值没有意义**。
- **只有 bwd_filter 的一族 engine（24、25、33、34、47、48、49、50）才有 `SPLIT_K[1..2048]`**：只有反向滤波（把 K 当规约维）才谈得上按 K 切分。
- 每个 engine 的旋钮数 0~9 个；`CUDNN_KNOB_TYPE_*` 一共 46 种（`cudnn_graph.h:3593-3639`），三个图合起来用到 17 种（fwd 图 16 种，多出来的 `SPLIT_K` 只在 bwd_filter）。
- runtime 编译组的旋钮里有 `KERNEL_CFG`——它就是"现场编译时选第几号 kernel 模板"（§11.10 里 engine 71 编译出的 `…convFwd_knob_21_128x128x16…` 即 `KERNEL_CFG=21`）。

#### 11.12.3 旋钮空间是稀疏的联合网格：不是每个点都能用

拿候选里旋钮最多的两个 engine 做"手工改旋钮"实验（`knob_sweep.py` → `knob_sweep.txt`）：

| 实验 | 结果 |
| --- | --- |
| engine 46：TILE_SIZE=4/STAGES=3 固定，KBLOCK 扫 0..5 | 只有 KBLOCK=2 能 finalize，其余 5 个值全部 `config finalize → NOT_SUPPORTED` |
| engine 46：TILE_SIZE=4/KBLOCK=2 固定，STAGES 扫 0..7 | 只有 STAGES=3 能 finalize |
| engine 46：heuristic 另外 3 个点 {1,3,2} / {5,2,6} / {5,3,4} | 全部能跑（稳定计时 0.127 / 0.099 / 0.139 ms，§11.12.4） |
| engine 46：自己拼的非候选点 {3,2,3}/{4,2,2}/{4,2,4}/{6,2,3}/{4,3,3} | **全部 NOT_SUPPORTED** |
| engine 48：TILEK 扫 0..3（其余保持候选 2 的值） | 只有 TILEK=2 能用；但候选 3 里 TILEK=0 又合法（搭配 TILE_SIZE=15、LDGC=0） |
| engine 28：TILE_SIZE 0..4、USE_TEX 0/1 | **全部可用**（这个 engine 的旋钮基本正交） |

结论：

- 旋钮取值不是连续区间，而是**一张离散、联合约束的网格**；网格上每个有效点 = 一份预编译好的 kernel 实例，无效点在 engine config finalize 时直接 `NOT_SUPPORTED`。
- **heuristic 返回的那几个点就是 cuDNN 已经验证过的有效点**；自己乱组合大概率直接报错（少数 engine 如 28 例外）。
- 所以"旋钮怎么选"的第一原则：**不要手选，用 heuristic 给的候选**。要在候选里再挑，就看 plan JSON 的 `engineId`+`knobChoices`（§11.9.2），或用 `CUDNN_HEUR_MODE_FALLBACK` 拿功能兜底候选——它的旋钮值会写成 `-1`（= "不指定，让 engine 自己定"，实测见 `engine_enum/engine_enum_modes_fwd.txt` 的 `HEUR_MODE=2` 段），并且会把 runtime 编译引擎也摆出来。

#### 11.12.4 换旋钮的实际收益（以及为什么小图上不明显）

同一 engine 内不同旋钮的稳定计时（200 迭代 × 3 次取最好，`knob_bench.py` → `knob_bench.txt`）：

| engine | 旋钮点 | ms/iter | 说明 |
| --- | --- | --- | --- |
| 28 | TILE_SIZE=3 | 0.0149 | 候选 #1，本组最快 |
| 28 | TILE_SIZE=4 | 0.0178 | heuristic top-1 |
| 28 | TILE_SIZE=2 | 0.0292 | |
| 28 | TILE_SIZE=0 | 0.0302 | |
| 28 | TILE_SIZE=1 | 0.0524 | 候选 #7，最慢（约 top-1 的 2.9 倍） |
| 46 | {4,2,3}（候选 6） | 0.0955 | top-1 |
| 46 | {5,2,6}（候选 10） | 0.0986 | |
| 46 | {1,3,2}（候选 9） | 0.1270 | |
| 46 | {5,3,4}（候选 13） | 0.1385 | 最慢（约 top-1 的 1.45 倍） |
| 48 | {15,0,0,0}（候选 3） | 0.0399 | |
| 48 | {2,2,0,1}（候选 2） | 0.0507 | |
| 3（FFT） | MULTIPLY=0 | 1.03 | |
| 3（FFT） | MULTIPLY=1 | **81.3** | 变体换成 `regular_fft_clip` → 慢约 79 倍 |
| 3（FFT） | MULTIPLY=2 | 1.70 | |

- 本机（RTX 5060 Laptop / WSL2）时钟漂移大：同一会话内排序稳定（engine 28：0.015~0.052 ms，约 3.5 倍；engine 46：1.45 倍），但绝对值跨会话能差 ±50%。所以 §11.9.3 里"单次 20 迭代"量出的 16 倍差要谨慎看（同一组点 200 迭代下约 3.5 倍）。**旋钮的收益在大图上才稳定体现**，cuDNN 自己也是用 GPUProfile + 形状在 heuristic 里算这笔账。
- 真正会"踩雷"的是换变体换出反例（FFT `MULTIPLY=1`）——所以更不该手工乱选旋钮。
- workspace 也是 engine config 的一部分（heuristic 的候选 JSON 里与 `knobChoices` 一起给出，§11.9.2）：改旋钮要连 workspace 一起看。

#### 11.12.5 怎么选：按推荐顺序

1. **什么都不做**：用 heuristic（`CUDNN_HEUR_MODE_A`/`INSTANT`）的 top-1——PyTorch 默认路径就是这条，实测命中的就是 top-1（§11.9.2）。
2. **要"钉死"某个实现**：从 heuristic 候选里挑一个点，或把 execution plan JSON（含 `cudnnVersion`/`smVersion`/`engineId`/`knobChoices`）存下来回放；跨版本/跨架构不保证复现。
3. **想要更多选择**：换 heuristic mode——A / INSTANT / B 对这张图给出的是**同一组 17 个候选**（实测 mode 0/1/3 的候选表逐行一致，见 `engine_enum/engine_enum_modes_fwd.txt`）；`FALLBACK`(2) 给 5 个功能兜底候选（含 runtime 编译 engine 71/39）。
4. **手拼旋钮**：只在"从 heuristic 已给出的点里挑"这个范围内是安全的（§11.12.3）。
5. PyTorch 侧不暴露旋钮，只有 `torch.backends.cudnn.benchmark`（= legacy 的 `Find...Ex` 自己试跑多个候选，§4.7.2/`logs/fe_conv_legacy_bench.log`）和 `benchmark_limit`。

一句话：**engine = 选算法（kernel 家族），旋钮 = 在该家族里选具体的预编译模板实例**。前者由 heuristic 排序挑，后者同样由 heuristic 给值；两者组合（engine config）才决定 workspace、布局转换、精度与最终 kernel。

#### 11.12.6 取 `KNOB_INFO` 的两个坑（都实测踩过）

1. **取回描述符数组前必须先自己 `cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR, …)`**：count 查询（`data=nullptr`）能正常返回个数，但直接拿一块空指针数组去取回，cuDNN 9.24 会在 `libcudnn_graph.so.9!cudnn::backend::Engine::get_internal` 里段错误（cuda-gdb 实测栈回到 `cudnnBackendGetAttribute`；见 `knob_info.cpp` 里的注释）。
2. **`CUDNN_ATTR_KNOB_INFO_TYPE` 要用 `CUDNN_TYPE_KNOB_TYPE` 查**（用 `CUDNN_TYPE_INT64` 查询会取不到值）。

### 11.13 heuristic（`libcudnn_heuristic.so.9`）的主要职责

**一句话：heuristic 是"选引擎的选择器"——把"这张图 × 这块 GPU × 这个 cuDNN 版本"下可用的 engine 空间，压缩成一张"已验证可 finalize、按预期性能排序"的短名单（每个候选带旋钮取值和 workspace），交给调用方去建 plan。它自己不发射 kernel、不做实测 benchmark、不保证精度。**

#### 11.13.1 做什么

| 职责 | 具体内容 | 证据 |
| --- | --- | --- |
| **筛** | 从 81/85/75 个全局 engine 里挑出本机可用的 17/12/16 个候选 | §11.9.2 |
| **配旋钮** | 给每个候选挑一组能通过 `EngineConfig` finalize 的旋钮取值（稀疏网格上的有效点） | §11.12.3：engine 46 上非候选点全部 `NOT_SUPPORTED` |
| **排序** | 按预期性能排 top-1；实测 top-1 与 PyTorch 实跑用的 engineId 一致（28 / 0 / 50） | §11.9.2 |
| **报告元信息** | numerical/behavior note、workspace、smVersion、GPUProfile（都写进候选的 plan JSON） | §11.9.1/§11.9.2 |
| **按模式给不同答案** | INSTANT=快决策树、B=神经网络、A=决策树（三者本机同候选）、FALLBACK=功能兜底（旋钮写 `-1`=不指定，含 runtime 编译引擎） | §11.12.5 + 下面 §11.13.3 实测 |

它的"聪明"来自设备属性 + 图特征（`OperationGraph::getHeurFeatures`、`isLargeTensorConv`，§4.7.1 的导入清单）+ 内置性能模型/KTM 预测（heuristic 注册的 3 类启发式 + 1 类 KTM 入口），**不是现场实测**——要实测得自己 benchmark（PyTorch 的 `cudnn.benchmark=True` 走的就是 legacy `Find...Ex`）。

#### 11.13.2 不做什么（边界）

| 不做什么 | 谁做 |
| --- | --- |
| 不发射 kernel | 三个 engine 库自己 `dlsym("cuLaunchKernel")` 后发射（§4.7.2） |
| 不保证候选之外的旋钮点可用 | 无效组合在 engine config finalize 时直接失败（§11.12.3） |
| 不保证 FALLBACK 候选的性能 | header 原文：`Functional fallback engines with no GPU performance guarantee` |
| 不保证数值/精度 | numerical note 只是标签；精度要自己校验（§11.9.3 的 maxerr 实测） |
| 不做实测性能比较 | 排序是"预期性能"；实测排序要 `cudnnFind...Ex` 或自己打表 |

注意候选（EngineConfig）本身还是用 graph 的实现构造的：heuristic 从 graph 导入 `EngineConfig::{ctor,finalize_internal}`、`ExecutionPlan::finalize_internal`、`EngineHeuristics::addEngineConfig`（§4.7.1）——**它只是"挑"，对象和内存都长在 graph 里**。

#### 11.13.3 代价实测（`engine_enum/heur_cost.cpp` → `run_heur_cost.sh` → `heur_cost.txt`）

方法：每种形状 × 每个模式各开新进程跑 3 次取中位数，进程内重复 4 轮（第 1 轮=冷，后 3 轮=热）。

| 形状（fwd, FP32） | 模式 | 候选数 | finalize | **首次取回候选** | 同进程再取回（缓存） |
| --- | --- | --- | --- | --- | --- |
| 4x3x64x64 → K=16 | INSTANT(0) | 17 | 0.04 ms | 161 ms | 0.18 ms |
| | B(1) | 17 | 0.05 ms | 135 ms | 0.19 ms |
| | FALLBACK(2) | 5 | 0.04 ms | 36 ms | 0.21 ms |
| | A(3) | 17 | 0.03 ms | 162 ms | 0.20 ms |
| 32x256x56x56 → K=256 | INSTANT(0) | 20 | 0.04 ms | 137 ms | 0.37 ms |
| | B(1) | 20 | 0.07 ms | 135 ms | 0.32 ms |
| | FALLBACK(2) | 5 | 0.03 ms | 0.9 ms | 0.17 ms |
| | A(3) | 20 | 0.04 ms | 134 ms | 0.31 ms |
| 8x1024x14x14 → K=1024 | INSTANT(0) | 19 | 0.05 ms | 140 ms | 0.29 ms |
| | B(1) | 19 | 0.05 ms | 132 ms | 0.30 ms |
| | FALLBACK(2) | 6 | 0.03 ms | 0.9 ms | 0.18 ms |
| | A(3) | 19 | 0.04 ms | 137 ms | 0.30 ms |

（表值是 3 次开新进程的中位数；复跑有 ±20% 抖动。）

读法：

- `finalize(heur)` 和 count 查询都是 **0.0x ms** 级；**重活全在"第一次取回候选数组"这一步**（cuDNN 内部惰性做枚举 + 排序），同进程第二次起 0.2~0.4 ms（结果被缓存）——所以"heuristic 贵不贵"取决于你问了几次、缓存没缓存。
- A/INSTANT/B 的首次代价都在 **0.13~0.2 s**，与形状弱相关；FALLBACK 便宜（大形状 ~0.9 ms；小图那次 36 ms 是个例外，未进一步定位）。
- header 说 mode B 比 INSTANT 贵 10~100×（CPU 开销），本机这三种卷积上**测不出这个差距**（冷/热都接近）——很可能要更大/更复杂的图才体现。
- 这 0.1~0.2 s 解释了 PyTorch 为什么把 plan 放进线程局部 LRU（§11.2）而不是每次卷积都重选。

#### 11.13.4 它挑的东西随形状变（职责的最直观体现）

同一台机器、同一份 cuDNN，只改形状：

| 形状（fwd） | 候选数 | top-1 | 它选了什么 |
| --- | --- | --- | --- |
| 4x3x64x64 → K=16 | 17 | engine 28 | FP32 `implicit_convolve_sgemm`（0 workspace，小图不值得做布局转换/降精度） |
| 32x256x56x56 → K=256 | 20 | engine 38 | TF32 xmma（105 MB workspace：算力优先） |
| 8x1024x14x14 → K=1024 | 19 | engine 39 | TF32 + `SPLIT_K_SLC=512`（连 split-K 切几份都是它算的） |

#### 11.13.5 接口与调用方一览

- graph API：`CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` + `CUDNN_ATTR_ENGINEHEUR_{OPERATION_GRAPH,MODE,RESULTS}`；取回结果前必须自建 engine config 数组（§11.9.5）。
- legacy 路径：`cudnnConvolutionForward` 等经 `cnn` 导入的 `getHeurFunc` 调同一个启发式（§4.7.1）。
- runtime 编译引擎用 `getTensorIRHeurFunc`。
- 注册方式：heuristic 静态构造器把 3 类启发式函数 + 1 类 KTM 预测函数注册进 graph，并用版本号（本机 92400）做握手校验（§4.7.1 反汇编）。

## 12. 文件索引

| 文件 | 内容 |
| --- | --- |
| `README.md` | 本文档 |
| `cudnn_imported_symbols.txt` | `libtorch_cuda.so` 引用的 83 个 cuDNN 符号（带 `@libcudnn.so.9` 版本后缀） |
| `cudnn_symbol_owners.txt` | 83 个符号 → 真实实现所在子库（graph / ops / cnn / adv）的逐行清单（§4.4） |
| `cudnn_graph_imports.txt` | ops / cnn / adv 三个库从 `graph` 导入的全部符号（按用途分组，§4.6） |
| `cudnn_engine_imports.txt` | 三个 engine 库 + `heuristic` 从 `graph` 导入的全部符号（83 / 192 / 39 / 28，按命名空间分组）、注册钩子清单与 graph↔插件 的版本握手反汇编记录（§4.7、§4.7.1） |
| `logs/fe_conv.log` | Conv2d 前反向（1.1k 次调用，纯 graph API） |
| `logs/fe_batchnorm.log` | BatchNorm2d（legacy Ex API） |
| `logs/fe_rnn.log` | LSTM 2 层（7.2MB，含 3208 次 `cudnnGetRNNWeightParams`） |
| `logs/fe_attn.log` | cuDNN SDPA（graph API） |
| `logs/fe_ctc4.log` | CTCLoss 命中 cuDNN 的形态（int32 1D labels） |
| `logs/fe_gs.log` | 私有 op `aten.cudnn_grid_sampler` |
| `logs/fe_affine.log` | 私有 op `aten.cudnn_affine_grid_generator` |
| `logs/fe_conv_legacy.log` | **legacy 卷积路径**（`TORCH_CUDNN_V8_API_DISABLED=1`，level 3）：公开 graph API 0 次，内部 `cudnnBackendExecuteInternal`×3 |
| `logs/fe_conv_legacy2.log` | 同上但 `CUDNN_LOGLEVEL_DBG=2`：只会看到 `_v7` 告警，用于说明日志级别的影响 |
| `logs/fe_conv_legacy_bench.log` | legacy 路径 + `benchmark=True`：8 个候选算法各查一次 workspace，3 次 `Find...Ex` |
| `probe_legacy_conv.py` | 强制 legacy 卷积路径的复现脚本 |
| `probe_legacy_bench.py` | legacy 路径 + `benchmark=True`（`cudnnFindConvolution*AlgorithmEx` 实测选算法），日志 `logs/fe_conv_legacy_bench.log` |
| `probe_handle.py` | 实测 handle 的脚本（地址/开销/内存里的 stream 偏移/NULL 与跨线程行为），输出见 `logs/handle_probe.out` |
| `logs/handle_probe.out` | `probe_handle.py` 的输出 |
| `graph_conv/graph_conv_backend.cpp` | **用 graph API 手写一次卷积**的最小 C++ 实现（自带 CPU 参考校验） |
| `graph_conv/run_graph_conv.sh` | 编译 + 运行 + 用日志复核调用次数 |
| `graph_conv/graph_conv.log` | 该 demo 的 cuDNN 调用日志（level 3） |
| `graph_conv/execute_async_demo.cpp` | 证明"execute 只是下发"的计时 demo（大卷积：host 侧 ~0.3 ms，GPU 还要 ~25 ms） |
| `graph_conv/run_execute_async.sh` | 编译并运行上面的计时 demo |
| `graph_conv/execute_async.log` | 计时 demo 的调用日志（含 variant pack 的 uid→指针映射） |
| `graph_conv/ld_libs.log` | `LD_DEBUG=libs` 输出：证明 pure-graph 程序在运行时才 dlopen 引擎库/heuristic（§4.7） |
| `legacy_conv/legacy_conv_backend.cpp` | **不用 graph API** 的等价实现（legacy 入口：描述符 + 算法选择 + `cudnnConvolutionForward`） |
| `legacy_conv/run_legacy_conv.sh` | 编译 + 运行（可带 `--all` / `--find`）+ 核对 `cudnnBackend*` 次数 |
| `legacy_conv/legacy_conv.log` | 默认（v7 启发式）路径的日志 |
| `legacy_conv/legacy_conv_all.log` | v7 请求全部 8 个候选的日志 |
| `legacy_conv/legacy_conv_find.log` | `Find...Ex` 实测选算法的日志 |
| `count_calls.py` | 按"入口记录"口径统计 cuDNN 日志调用次数的脚本（原理见 §8.1） |
| `run_per_op_all.sh` | 把 `/home/liquan/cudnn_per_op.py` 的每个 case 各跑一遍、自动提取 case 列表、保存日志并汇总计数（§5.6） |
| `logs_per_op/` | 上述 7 个 case 的 cuDNN level-3 日志 + `.stdout` + `summary.txt`（§5.6） |
| `dlsym_probe/dlsym_who.c` | `LD_PRELOAD` 探针：包装 `dlsym()` 并记录调用方 DSO（回答"谁发射 kernel"，§4.7.2） |
| `dlsym_probe/{graph_conv,legacy_conv}.dlsym.log` | 两个 demo 的 driver API 解析记录（引擎自己解析 `cuLaunchKernel`，graph 不解析） |
| `launch_probe/kernel_trace.py` | 用 CUPTI(torch.profiler) 抓每个 case 的 kernel 发射（名字/grid/block/smem/regs/stream/耗时）并标注归属库（§4.7.3） |
| `launch_probe/kernels_<case>.{txt,json}` | 各 case 的 kernel 发射表与 chrome trace（`conv_fwd`/`conv_bwd` 为前后向拆分） |
| `launch_probe/engine_kernel_map.txt` | `engine_id` ↔ 实际 kernel 的对照（同进程同时开 cuDNN 日志与 CUPTI 得到） |
| `launch_probe/launch_trace.c` | 驱动符号替换的尝试（`LD_PRELOAD` 包 `dlsym`），本机受限于 `cuGetProcAddress` 路径（§4.7.3 说明） |
| `engine_enum/engine_enum.cpp` | **枚举一张卷积图上的全部 engine**：打印每个 engine 的 numerical/behavior note、布局、workspace，并让 heuristic 排序、从 execution plan 的 JSON 里取 engineId 与旋钮（§11.9） |
| `engine_enum/engine_enum_{fwd,bwd_data,bwd_filter}.{txt,log}` | 三种卷积图的 engine 普查表（全局数/可用数/标签/候选表）与 level-3 日志 |
| `engine_enum/runner.cpp` + `librunner.so` | 把某个 heuristic 候选真跑起来（ctypes 调用），返回 JSON/耗时/与 CPU 参考的 maxerr，供 CUPTI 抓 kernel（§11.9.3） |
| `engine_enum/trace_engines.py` | 在 `torch.profiler` 里逐个候选执行并抓 kernel、标注归属库；输出 `trace_all.txt`、`kernels_cand*.txt`、`engines_kernels.txt`、`trace_cand*.json` |
| `engine_enum/README.txt` / `run_engine_enum.sh` | 该目录的方法与结论摘要 / 一键复现脚本 |
| `engine_enum/candidate_json_*.json` | heuristic 各候选的 execution plan JSON（含 engineId、旋钮、GPUProfile、张量描述） |
| `engine_enum/knob_info.cpp` → `knob_info_{fwd,bwd_data,bwd_filter}.txt` | **逐个 engine 列出旋钮**：类型 / `[min..max]` / 步长（§11.12.2）；配套 `knob_info_pc.cpp`（`RETR_CREATE=1` 走"预创建描述符"路径）复现 §11.12.6 的段错误条件 |
| `engine_enum/knob_sweep.py` → `knob_sweep.txt`、`knob_kernels.txt` | **手工旋钮网格实测**：engine 28/1/3/6/46/48 的逐点执行结果、无效点报错、旋钮→kernel 模板对照（§11.12.3/§11.12.4） |
| `engine_enum/knob_bench.py` → `knob_bench.txt` | 同一批"engine+旋钮"点的稳定计时（200 迭代 × 3 取最好，§11.12.4） |
| `engine_enum/engine_enum_modes_fwd.txt` | fwd 图在 4 种 heuristic mode（INSTANT/B/A/FALLBACK）下的候选对照（§11.12.5） |
| `engine_enum/heur_cost.cpp` + `run_heur_cost.sh` → `heur_cost.txt` | **heuristic 的 CPU 代价实测**：3 种形状 × 4 种模式，分段计时（finalize / 首次取回候选 / 缓存后）+ 候选数（§11.13.3） |
| `engine_enum/trace_engine_global.py` | 按**全局 engine 编号**执行 + CUPTI 抓 kernel + 报 plan finalize 耗时（对比预编译 / runtime 编译引擎，§11.10） |
| `engine_enum/kernels_engine{28,71}.txt`、`trace_engine{28,71}.json` | precompiled（28）与 runtime_compiled（71）引擎的 kernel 表与 chrome trace |
| `engine_enum/engine71_runtime.log` | engine 71（runtime 编译引擎）执行的 cuDNN level-3 日志 |
