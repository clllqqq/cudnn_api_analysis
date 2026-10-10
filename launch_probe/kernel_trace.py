#!/usr/bin/env python3
"""用 CUPTI(torch.profiler / Kineto) 抓算子执行里的 kernel 发射记录。

回答的问题：cuDNN 引擎到底"怎么"把 kernel 发出去——
  看到了 kernel 名字、grid/block、动态共享内存、寄存器数、stream、耗时。
并把每个 kernel 名字归属到 cuDNN 子库（在 .so 里做字符串搜索）。

用法:
  kernel_trace.py conv                # 跑 /home/liquan/cudnn_per_op.py 的某个 case
  kernel_trace.py conv rnn attn       # 顺序跑多个 case
  TARGET=/path/to/other.py kernel_trace.py <arg>
输出: <这里>/kernels_<case>.txt 与 kernels_<case>.json
"""
import glob, json, os, subprocess, sys

TARGET = os.environ.get("TARGET", "/home/liquan/cudnn_per_op.py")
HERE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = os.environ.get(
    "CUDNN_LIB",
    "/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia/cudnn/lib",
)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402


def lib_index():
    libs = {}
    for path in sorted(glob.glob(os.path.join(LIBDIR, "libcudnn_*.so.9"))):
        with open(path, "rb") as fh:
            libs[os.path.basename(path).replace(".so.9", "")] = fh.read()
    return libs


def _demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout
    return out.splitlines()


def attribute(name, libs, demangled=None):
    """在 .so 里搜 kernel 名字：先去 mangle，再按 :: 逐级截短。"""
    if "cudnn_generated" in name or "nvrtc" in name:
        return "runtime JIT(引擎现场编译)"
    _skip = ("at::", "c10::", "cublas", "(anonymous", "cutlass::Kernel2")
    if name.startswith(_skip) or name.startswith("void at::"):
        return "?"
    cands = []
    full = (demangled or name).split("(")[0]
    base = full.split("<")[0].strip().removeprefix("void ").strip()
    parts = base.split("::")
    for i in range(len(parts)):          # 后缀式候选：永远带上函数名
        cands.append("::".join(parts[i:]))
    for c in cands:
        if len(c) < 12 or c.startswith("at::") or c.startswith("cublas") or c.startswith("cutlass::Kernel2"):
            continue
        b = c.encode()
        for lib, blob in libs.items():
            if b in blob:
                return lib
    return "?"


def run_case(case, libs):
    src = open(TARGET).read()
    argv = sys.argv[:1] + [case]
    g = {"__name__": "__main__", "__file__": TARGET}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        old = sys.argv
        sys.argv = argv
        try:
            exec(compile(src, TARGET, "exec"), g)
        finally:
            sys.argv = old
        torch.cuda.synchronize()

    trace = os.path.join(HERE, f"kernels_{case}.json")
    prof.export_chrome_trace(trace)
    events = json.load(open(trace))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    dem = _demangle([e["name"] for e in kernels])

    out = os.path.join(HERE, f"kernels_{case}.txt")
    with open(out, "w") as fh:
        fh.write(f"# {case}: {len(kernels)} 个 kernel 发射（CUPTI，torch.profiler）\n")
        fh.write(f"# target={TARGET} case={case}\n")
        fh.write(f"# {'t_us':>10s} {'kernel':72s} {'grid':16s} {'block':16s} {'smem':>8s} {'regs':>5s} {'stream':>6s} {'us':>9s}  归属库\n")
        for e, dname in zip(kernels, dem):
            a = e.get("args", {})
            g_, b_ = a.get("grid"), a.get("block")
            fh.write(
                f"{e.get('ts', 0):10.1f} {e['name'][:72]:72s} {str(g_):16s} {str(b_):16s} "
                f"{str(a.get('shared memory')):>8s} {str(a.get('registers per thread')):>5s} "
                f"{str(e.get('tid')):>6s} {e.get('dur', 0):9.1f}  {attribute(e['name'], libs, dname)}\n"
            )
    return out, len(kernels)


if __name__ == "__main__":
    cases = sys.argv[1:] or ["conv"]
    libs = lib_index()
    for case in cases:
        out, n = run_case(case, libs)
        print(f"{case}: {n} kernels -> {out}")
