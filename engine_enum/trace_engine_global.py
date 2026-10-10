#!/usr/bin/env python3
"""按全局 engine 编号执行并用 CUPTI 抓 kernel（对比预编译 / runtime 编译引擎）。
用法: python trace_engine_global.py 28 71 0 6
输出: kernels_engine<id>.txt
"""
import ctypes, importlib.util, json, os, sys

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "kernel_trace", "/home/liquan/cudnn_api_analysis/launch_probe/kernel_trace.py")
kt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kt)

lib = ctypes.CDLL(os.path.join(HERE, "librunner.so"))
lib.run_engine_global.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double)]
lib.run_engine_global.restype = ctypes.c_int

libs = kt.lib_index()
for eid in [int(x) for x in sys.argv[1:]] or [28, 71]:
    buf = ctypes.create_string_buffer(8192)
    pm = ctypes.c_double(0); ms = ctypes.c_double(0); err = ctypes.c_double(0)
    rc = lib.run_engine_global(eid, 1, buf, len(buf), ctypes.byref(pm), ctypes.byref(ms), ctypes.byref(err))
    if rc:
        print(f"engine {eid}: rc={rc}（需要旋钮或本机不支持）")
        continue
    iters = 20
    trace = os.path.join(HERE, f"trace_engine{eid}.json")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        lib.run_engine_global(eid, iters, None, 0, ctypes.byref(pm), ctypes.byref(ms), ctypes.byref(err))
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    events = json.load(open(trace))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    dem = kt._demangle([e["name"] for e in kernels])
    with open(os.path.join(HERE, f"kernels_engine{eid}.txt"), "w") as fh:
        fh.write(f"# 全局 engine {eid}: plan finalize={pm.value:.2f} ms  {ms.value:.4f} ms/iter  "
                 f"maxerr={err.value:.2e}  kernels={len(kernels)}\n")
        for e, dname in zip(kernels, dem):
            a = e.get("args", {})
            fh.write(f"  {e['name'][:96]:96s} grid={str(a.get('grid')):16s} block={str(a.get('block')):16s} "
                     f"smem={str(a.get('shared memory')):>7s} regs={str(a.get('registers per thread')):>4s} "
                     f"{e.get('dur', 0):8.1f}us  {kt.attribute(e['name'], libs, dname)}\n")
    print(f"engine {eid}: plan={pm.value:.2f} ms  {ms.value:.4f} ms/iter  maxerr={err.value:.2e}  kernels={len(kernels)}")
    seen = []
    for e, dname in zip(kernels, dem):
        if e["name"] not in seen:
            seen.append(e["name"])
            print(f"    {e['name'][:92]:92s} {kt.attribute(e['name'], libs, dname)}")
