# dlsym 探针：谁在解析 CUDA driver API（谁真正发射 kernel）
#
# 背景（见 ../README.md §4.7.2）：
#   graph 对引擎库 0 符号引用，那么 kernel 由谁发射？
# 探针：
#   LD_PRELOAD 一个包装 dlsym() 的 .so，记录每次 dlsym("cuXxx") 的调用方 DSO
#   （通过 dladdr(__builtin_return_address(0)) 得到调用方）。
#   注意：不能只包装 cuLaunchKernel —— 各 DSO 是对 libcuda.so.1 的句柄做 dlsym，
#   不走全局符号查找，所以必须从 dlsym 入口拦。
#
# 复现：
#   gcc -shared -fPIC -O2 dlsym_who.c -o /tmp/dlsym_who.so -ldl
#   NV=/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia
#   LD_PRELOAD=/tmp/dlsym_who.so LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cu13/lib \
#       ../graph_conv/graph_conv_backend  2> graph_conv.dlsym.log
#   LD_PRELOAD=/tmp/dlsym_who.so LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cu13/lib \
#       ../legacy_conv/legacy_conv_backend 2> legacy_conv.dlsym.log
#
# 结论（两份日志一致）：
#   libcudnn_engines_precompiled.so.9 自己解析并调用：
#     cuInit / cuDriverGetVersion / cuDeviceGet / cuDeviceGetAttribute
#     cuLinkCreate_v2 / cuLinkAddData_v2 / cuLinkComplete / cuLinkDestroy   ← JIT 链接
#     cuModuleLoadData / cuModuleLoadDataEx / cuModuleLoadFatBinary
#     cuModuleGetFunction / cuFuncSetAttribute / cuFuncGetAttribute / cuModuleUnload
#     cuLaunchKernel / cuLaunchKernelEx                                    ← 真正发射
#     cuCtxPushCurrent / cuCtxPopCurrent / cuTensorMapEncodeTiled ...
#   libcudnn_graph.so.9 只解析：
#     cuDriverGetVersion / cuDeviceGet / cuDeviceGetAttribute / cuGetProcAddress_v2
#     cuModuleLoadData / cuModuleGetFunction / cuFuncSetAttribute
#     cuOccupancy*                                                          ← 只做容量查询
#     **没有 cuLaunchKernel / cuLinkCreate** —— graph 不发射 kernel
