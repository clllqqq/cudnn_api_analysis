#!/usr/bin/env python3
"""对卷积图上 heuristic 的每个候选 engine 逐个执行并用 CUPTI 抓 kernel。

- 用 ctypes 调 librunner.so（同一进程内跑 cuDNN graph），torch.profiler(CUPTI) 能看到它发的 kernel；
- 每个候选：先 warmup 1 次，再在 profiler 里跑 N 次，导出 chrome trace；
- kernel 名在 cuDNN 各 .so 里做字符串搜索来标注归属库。
用法: python trace_engines.py [起始候选] [结束候选] [每候选迭代数]
输出: trace_cand<k>.json / kernels_cand<k>.txt，汇总在 engines_kernels.txt
"""
import ctypes, glob, importlib.util, json, os, re, subprocess, sys

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = "/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia/cudnn/lib"

spec = importlib.util.spec_from_file_location(
    "kernel_trace", "/home/liquan/cudnn_api_analysis/launch_probe/kernel_trace.py")
kt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kt)

lib = ctypes.CDLL(os.path.join(HERE, "librunner.so"))
lib.run_cand.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.run_cand.restype = ctypes.c_int
lib.num_cands.restype = ctypes.c_int


def field(json_str, key):
    m = re.search(r'"%s":([^,}]*)' % re.escape(key), json_str)
    return m.group(1) if m else "?"


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else lib.num_cands()
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    libs = kt.lib_index()
    summary = []
    for cand in range(lo, hi):
        buf = ctypes.create_string_buffer(8192)
        ms = ctypes.c_double(0)
        err = ctypes.c_double(0)
        rc = lib.run_cand(cand, 1, buf, len(buf), ctypes.byref(ms), ctypes.byref(err))
        if rc:
            print(f"cand {cand}: warmup 失败 rc={rc}")
            summary.append((cand, "?", "?", None, None, "RUN_FAILED"))
            continue
        js = buf.value.decode()
        engine_id = field(js, "engineId")
        knobs = re.search(r'"knobChoices":\{([^}]*)\}', js)
        knobs = knobs.group(1) if knobs else "-"

        trace = os.path.join(HERE, f"trace_cand{cand}.json")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            lib.run_cand(cand, iters, None, 0, ctypes.byref(ms), ctypes.byref(err))
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace)

        events = json.load(open(trace))["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        dem = kt._demangle([e["name"] for e in kernels])
        rows = []
        for e, dname in zip(kernels, dem):
            a = e.get("args", {})
            rows.append((e["name"], a.get("grid"), a.get("block"),
                         a.get("shared memory"), a.get("registers per thread"),
                         kt.attribute(e["name"], libs, dname), e.get("dur", 0)))
        with open(os.path.join(HERE, f"kernels_cand{cand}.txt"), "w") as fh:
            fh.write(f"# heuristic 候选 #{cand}: engineId={engine_id}  "
                     f"{ms.value:.4f} ms/iter  maxerr={err.value:.2e}\n")
            fh.write(f"# 旋钮 {knobs}\n")
            for r in rows:
                fh.write(f"  {r[0][:96]:96s} grid={str(r[1]):16s} block={str(r[2]):16s} "
                         f"smem={str(r[3]):>7s} regs={str(r[4]):>4s} {r[6]:8.1f}us  {r[5]}\n")
        print(f"cand {cand:2d} engineId={engine_id:>4s} {ms.value:8.4f} ms  maxerr={err.value:.2e}  "
              f"kernels={len(kernels)} -> kernels_cand{cand}.txt")
        for r in rows:
            print(f"      {r[0][:88]:88s} {r[5]}")
        summary.append((cand, engine_id, knobs, ms.value, err.value,
                        "; ".join(r[0] for r in rows)))

    with open(os.path.join(HERE, "engines_kernels.txt"), "w") as fh:
        fh.write("# 候选 engine -> 实际发射的 kernel（CUPTI，torch.profiler）\n")
        for c, e, k, ms, err, ks in summary:
            fh.write(f"#{c}\tengine={e}\t{ms}\tms\tmaxerr={err}\tknobs={k}\n\t{ks}\n")
    print("\n汇总写入 engines_kernels.txt")


if __name__ == "__main__":
    main()
