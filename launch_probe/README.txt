# 观察 cuDNN 引擎如何发射 kernel —— 工具与结论（对应 ../README.md §4.7.2 / §4.7.3）
#
# 一、能用的：CUPTI / torch.profiler（推荐）
#     kernel_trace.py <case> [case2 ...]
#        在 torch.profiler(ProfilerActivity.CUDA) 里 exec /home/liquan/cudnn_per_op.py 的指定 case，
#        导出 chrome trace，并把 kernel 名在 cuDNN 各子库(.so)里做字符串搜索来标注归属。
#        输出：kernels_<case>.txt（表格）+ kernels_<case>.json（可用 chrome://tracing 打开看时间线）
#     复现：
#        /home/liquan/venvs/torch/bin/python kernel_trace.py conv batchnorm rnn attention_forced
#     实测（conv，与 cuDNN 日志的 engine_id 对齐）：
#        fwd   engine_id=28  implicit_convolve_sgemm<...,1024,5,5,3,3,3,1,...>
#                            grid=[512,1,1] block=[8,8,1] smem=2304 regs=63   (engines_precompiled)
#        dgrad engine_id=0   dgrad2d_grouped_direct_kernel<...>
#                            grid=[48,1,1] block=[1024,1,1] regs=64           (engines_precompiled)
#        wgrad engine_id=50  nchwToNhwc×2 → cutlass__5x_cudnn::Kernel<...ImplicitGemmConvolution...>
#                            grid=[1,1,4] block=[128,1,1] smem=81920 regs=106 → reduction → nhwcToNchw
#        batchnorm           cudnn::bn_fw_tr_1C11_singleread<float,512,...>
#                            grid=[16,1,1] block=[512,1,1] smem=32912           (libcudnn_ops)
#        attention_forced    cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_1_128x64...
#                            grid=[1,8,2] block=[128,1,1] smem=40960 regs=234   (runtime 现场编译引擎)
#
# 二、不行的：驱动符号替换（launch_trace.c）
#     LD_PRELOAD 包装 dlsym()，把引擎 dlsym 到的 cuLaunchKernel(_ptsz/Ex) 换成自己的包装。
#     机制本身可用（用一个自测程序验证过），但本机看不到引擎的发射，原因：
#       1) cuDNN 引擎取发射入口走的是 cuGetProcAddress_v2，而不是 dlsym 返回的那份；
#       2) WSL 版 libcuda 内部也会经 PLT 调同名符号，一旦被 preload 抢占就与本库互相递归
#          （ping-pong），因此不能包装 cuGetProcAddress*。
#     所以 launch_trace.c 保留为"尝试记录"：它只包装 launch 相关的 4 个符号，跑起来安全，
#     但输出为空。真正的答案用 CUPTI 那条路。
#
# 三、Nsight Systems
#     /usr/local/cuda/bin/nsys profile --trace=cuda ./demo  ->  "does not contain CUDA trace data"
#     （WSL 下 CUPTI 注入受限；torch.profiler 路线正常，因为它是在进程内使用 CUPTI。）
