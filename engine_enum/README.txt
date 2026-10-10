# 枚举 / 解剖 cuDNN 的 engine —— 工具与结论（对应 ../README.md §11.9）
#
# 一、工具
#   engine_enum.cpp / engine_enum
#       建一张卷积图（fwd / bwd_data / bwd_filter 三种，形状取自 cudnn_per_op.py 的 conv case：
#       N=4 C=3 H=W=64，K=16 R=S=3，pad=1 stride=1），然后：
#         · 读 CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT，逐个全局编号建
#           CUDNN_BACKEND_ENGINE_DESCRIPTOR，打印 numerical_note / behavior_note /
#           layout / SM target，并用 engine config 读 workspace；
#         · 跑 heuristic（mode A），把每个候选做成 execution plan，从
#           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION 里取 engineId 与 knobChoices。
#
#   runner.cpp -> librunner.so
#       把某个 heuristic 候选真的执行起来（ctypes 调用），供 CUPTI 抓 kernel：
#       int run_cand(cand, iters, json_out, cap, double* ms, double* err)
#       —— 返回 maxerr（与 CPU 参考卷积对比）保证"算的还是同一个卷积"。
#
#   trace_engines.py
#       在 torch.profiler(ProfilerActivity.CUDA) 里逐个候选调 run_cand，
#       导出 chrome trace，并把 kernel 名在 cuDNN 各 .so 里做字符串搜索标注归属库。
#       用法: python trace_engines.py [起始] [结束] [每候选迭代数]
#
#   knob_info.cpp / knob_info       逐个 engine 列旋钮：类型 + [min..max] + 步长（§11.12.2）
#       用法: ./knob_info fwd|bwd_data|bwd_filter [起始 engine] [结束 engine]
#       输出: knob_info_{fwd,bwd_data,bwd_filter}.txt
#       注意：取回 KNOB_INFO 描述符数组前必须先自己 CreateDescriptor(KNOB_INFO)，
#             否则 cuDNN 9.24 在 libcudnn_graph!Engine::get_internal 里段错误；
#             KNOB_INFO_TYPE 要用 CUDNN_TYPE_KNOB_TYPE 查。
#
#   knob_sweep.py                   手工指定"engine + 旋钮"执行（ctypes 调
#       librunner.so 的 run_engine_global_knobs），验证旋钮网格的稀疏性：
#       输出 knob_sweep.txt（每个点的 rc/plan_ms/ms/maxerr）与
#       knob_kernels.txt（旋钮 → kernel 模板实例）。§11.12.3/§11.12.4
#
#   knob_bench.py                   对"engine + 旋钮"点做稳定计时
#       （200 迭代 × 3 次取最好），输出 knob_bench.txt。§11.12.4
#
#   engine_enum.cpp + HEUR_MODE=0/1/2/3   对比 4 种 heuristic 模式给出的候选
#       （输出 engine_enum_modes_fwd.txt；本机 A/INSTANT/B 同候选，FALLBACK 不同）。§11.12.5
#
#   heur_cost.cpp / run_heur_cost.sh      heuristic 的 CPU 代价分段计时（§11.13.3）
#       用法: ./heur_cost <mode 0..3> [reps]   （形状用 RUNNER_N/C/HW/K 改）
#       输出: 每轮 set/finalize/取回候选 三段耗时 + 候选数；run_heur_cost.sh 汇总到 heur_cost.txt
#       实测: finalize ~0.04ms、count ~0.15ms；重活全在"首次取回候选"（A/B/INSTANT 0.13~0.2s，
#             FALLBACK 0.8~31ms），同进程第二次起 0.2~0.4ms（缓存）。
#
#   run_engine_enum.sh  —— 一键复现全部输出
#
# 二、结论（本机 RTX 5060 Laptop / sm_120 / cuDNN 9.24.0）
#   1) 全局 engine 数（= 描述符能枚举出的数量，含不适配本机架构的）：
#        fwd 81（可用 25）/ bwd_data 85（可用 33）/ bwd_filter 75（可用 31）
#      "可用" = engine 描述符能 finalize；失败的是 ARCH_MISMATCH（别的架构的 kernel）或 NOT_SUPPORTED。
#   2) engine 的自我标签（numerical_note / behavior_note）：
#        fwd: TensorCore 16、Winograd 1、FFT 2、RUNTIME_COMPILATION 5、STRICT_NAN_PROP 1
#        bwd_data: TensorCore 22、FFT 3、Winograd 1、非确定性 1
#        bwd_filter: TensorCore 24、FFT 1、非确定性 1
#   3) heuristic(mode A) 只排一小撮：fwd 17 个、bwd_data 12 个、bwd_filter 16 个候选；
#      三个图的 top-1 分别落在 engine 28 / 0 / 50 —— 与 PyTorch 实跑时日志里的
#      engine_id=28 / 0 / 50 完全一致。
#   4) 17 个候选逐个真跑 + CUPTI 抓 kernel（20 次迭代平均，形状 4x3x64x64 -> 4x16x64x64）：
#        候选  engineId  主 kernel                                        精度(maxerr)  库
#         0/1/7   28     implicit_convolve_sgemm（模板参数随 TILE_SIZE 变）  2.4e-06     engines_precompiled
#         2/3     48     nchwToNhwc → xmma tf32(wo_smem) → nhwcToNchw      3.3e-03     engines_precompiled
#         4       1      kern_precompute_indices + precomputed_convolve_sgemm 2.4e-06   engines_precompiled
#         5       6      winogradForward{Data,Filter}4x4 + gemm + Output   1.4e-05     engines_precompiled
#         6/9/10/13 46   nchwToNhwc → cutlass tensorop tf32 → nhwcToNchw  3.3e-03     engines_precompiled
#         8       0      conv2d_grouped_direct_kernel（无 workspace）        2.4e-06     engines_precompiled
#         11      36     xmma tf32 indexed 128x128x16                    3.3e-03     engines_precompiled
#         12      38     xmma tf32 nchw 直出（输出不转布局）                3.3e-03     engines_precompiled
#         14      35     cutlass tensorop tf32 analytic 128x64           3.3e-03     engines_precompiled
#         15      3     DSE::regular_fft_*/vector_fft + cgemm（FFT 引擎）  5.2e-06     engines_precompiled
#         16      4     fft2d_r2c/c2r_32x32 + cgemm（FFT 引擎，588 次发射） 4.8e-06     engines_precompiled
#      结论：同一张图、同一个正确结果，引擎之间的差别 = 换 kernel（算法）+ 换旋钮（同一 kernel
#      的不同 tile/stage 配置）+ 换布局（NCHW↔NHWC 转换）+ 换精度（FP32 / TF32）+ 换 workspace。
#   5) 全部 kernel 都落在 libcudnn_engines_precompiled.so.9；带 RUNTIME_COMPILATION 标签的
#      引擎（全局表里的 5/71/72/79/80）这次没进 heuristic 的候选，它们才是 NVRTC 现场编译
#      （SDLA 那种 `cudnn_generated_*` kernel，见 ../launch_probe/kernels_attention_forced.txt）。
#
# 四、precompiled vs runtime_compiled（§11.10）
#   trace_engine_global.py <engine_id> [...]   按全局 engine 编号执行 + CUPTI 抓 kernel + 报 plan 耗时
#     python trace_engine_global.py 28 0 6 71
#   实测（同一张 4x3x64x64->4x16x64x64 图）：
#     engine 0 / 28 / 1 / 2 / 6 / 48（precompiled）  plan finalize 0.02 ~ 0.21 ms
#     engine 71（RUNTIME_COMPILATION）               plan finalize 19.9 ~ 39.7 ms（首次冷启动 478 ms）
#        kernel: cudnn_generated_fort_native_sm80_convFwd_knob_21_128x128x16_4_0
#     代价差 3 个数量级，钱花在 plan（现场生成 kernel），不在 execute。
#   runner.cpp 里 run_engine_global(engine_id, iters, json, cap, plan_ms, ms, err) 可直接复用。
#
# 五、engine 编号的语义（§11.11）
#   engine_enum.cpp 支持用环境变量改形状：RUNNER_N / RUNNER_C / RUNNER_HW / RUNNER_K
#   实测：N=4/8/9/16、C=3/64、64x64 与 224x224、K=3/16/64 —— fwd 图恒为 81 个 engine，
#         只有"能 finalize 的个数"偶尔变（K=3 时 24 个）。fwd/bwd_data/bwd_filter = 81/85/75。
#   结论：编号 = 该图引擎表的下标（0..count-1 稠密），count 由图的结构+设备+cuDNN 版本决定，
#         与形状无关；同一 (版本, 设备, 图) 下跨进程稳定；不同图里同一编号不是同一个引擎
#         （#28 在 fwd 是 FP32 sgemm/ws=0，在 bwd_data 是 TENSOR_CORE/ws=1313040）。
#
# 三、方法上踩过的坑（复现时注意）
#   1) 只读属性的 count 查询有两种口径：多数属性要 data=nullptr；而 ENGINEHEUR_RESULTS
#      必须给非空指针，且取回时必须先自己创建好 descriptor 数组（否则 cuDNN 会往缓冲区里
#      写指针，直接踩内存）。
#   2) heuristic 返回的 engine config 是"借用"句柄，不能再 GetAttribute 它的 ENGINE /
#      KNOB_CHOICES（BAD_PARAM 或崩溃）；要拿 engineId/旋钮就走 execution plan 的 JSON。
#   3) 读 EXECUTION_PLAN_WORKSPACE_SIZE 时不要把 count 和 data 用同一个变量接，
#      否则拿到的是 1（count）而不是 size —— 后续 execute 会因 workspace 太小而失败。
#   4) 旋钮网格是稀疏的：实测 engine 46 的 (TILE_SIZE,KBLOCK,STAGES) 里，只有 heuristic
#      给出的 4 个点能 finalize（{4,2,3}/{1,3,2}/{5,2,6}/{5,3,4}），其余组合一律
#      config finalize → CUDNN_STATUS_NOT_SUPPORTED；engine 48 同理（只有候选 2/3 两点）。
#      而 engine 28 的 TILE_SIZE[0..4] 全可用 —— 旋钮能否自由组合取决于 engine（§11.12.3）。
#   5) 取 KNOB_INFO 描述符数组前必须先自己 CreateDescriptor(KNOB_INFO)（否则段错误）；
#      且 KNOB_INFO_TYPE 要用 CUDNN_TYPE_KNOB_TYPE 查（§11.12.6）。
