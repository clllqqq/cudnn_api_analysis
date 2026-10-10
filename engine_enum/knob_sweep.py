#!/usr/bin/env python3
"""手工指定 engine + 旋钮，实测同一 engine 下不同旋钮的耗时/精度/kernel。
用法: python knob_sweep.py            （输出 knob_sweep.txt / knob_kernels.txt）
"""
import ctypes, json, os, sys

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "librunner.so"))
lib.run_engine_global_knobs.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.run_engine_global_knobs.restype = ctypes.c_int

# cudnnBackendKnobType_t
K = dict(SPLIT_K=0, TILE_SIZE=2, USE_TEX=3, KBLOCK=5, LDGA=6, MULTIPLY=11, SPLIT_K_BUF=12,
         TILEK=13, STAGES=14, REDUCTION_MODE=15, SPLIT_K_SLC=17, IDX_MODE=18, LDGC=22,
         SPECFILT=23, KERNEL_CFG=24, WORKSPACE=25, STREAM_K=38)

NAMES = {v: k for k, v in K.items()}


def run(eid, knobs, iters=20):
    types = (ctypes.c_int * len(knobs))(*[K[n] for n, _ in knobs])
    vals = (ctypes.c_longlong * len(knobs))(*[v for _, v in knobs])
    buf = ctypes.create_string_buffer(16384)
    pm, ms, err = (ctypes.c_double(0) for _ in range(3))
    rc = lib.run_engine_global_knobs(eid, types, vals, len(knobs), iters, buf, len(buf),
                                     ctypes.byref(pm), ctypes.byref(ms), ctypes.byref(err))
    return rc, pm.value, ms.value, err.value, buf.value.decode(errors="replace")


def profile_kernels(eid, knobs, iters=10):
    """跑一遍并用 profiler 抓 kernel 名（导出临时 chrome trace 再解析）。"""
    types = (ctypes.c_int * len(knobs))(*[K[n] for n, _ in knobs])
    vals = (ctypes.c_longlong * len(knobs))(*[v for _, v in knobs])
    pm, ms, err = (ctypes.c_double(0) for _ in range(3))
    tmp = os.path.join(HERE, "_knob_tmp_trace.json")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        rc = lib.run_engine_global_knobs(eid, types, vals, len(knobs), iters, None, 0,
                                         ctypes.byref(pm), ctypes.byref(ms), ctypes.byref(err))
        torch.cuda.synchronize()
    if rc:
        return rc, pm.value, ms.value, err.value, []
    prof.export_chrome_trace(tmp)
    ev = json.load(open(tmp))["traceEvents"]
    names, seen = [], set()
    for e in ev:
        if e.get("cat") == "kernel" and e["name"] not in seen:
            seen.add(e["name"])
            names.append(e["name"])
    try:
        os.remove(tmp)
    except OSError:
        pass
    return rc, pm.value, ms.value, err.value, names


CASES = []
def add(eid, knobs, tag):
    CASES.append((eid, knobs, tag))

for t in range(5):
    add(28, [("TILE_SIZE", t), ("USE_TEX", 0)], f"eng28 TILE_SIZE={t}")
add(28, [("TILE_SIZE", 4), ("USE_TEX", 1)], "eng28 TILE_SIZE=4 USE_TEX=1")
for t in range(5):
    add(1, [("TILE_SIZE", t), ("USE_TEX", 0)], f"eng1 TILE_SIZE={t}")
for t in range(3):
    add(3, [("MULTIPLY", t)], f"eng3(FFT) MULTIPLY={t}")
for w in range(3):
    add(6, [("WORKSPACE", w)], f"eng6(Winograd) WORKSPACE={w}")
base48 = [("TILE_SIZE", 2), ("LDGA", 0), ("TILEK", 2), ("STAGES", 0), ("LDGC", 1)]
add(48, base48, "eng48 基线(=heuristic 候选2)")
add(48, [("TILE_SIZE", 15), ("LDGA", 0), ("TILEK", 0), ("STAGES", 0), ("LDGC", 0)], "eng48 基线2(=heuristic 候选3)")
for t in range(4):
    add(48, [("TILE_SIZE", 2), ("LDGA", 0), ("TILEK", t), ("STAGES", 0), ("LDGC", 1)], f"eng48 TILEK={t}")
for s in range(8):
    add(48, [("TILE_SIZE", 2), ("LDGA", 0), ("TILEK", 2), ("STAGES", s), ("LDGC", 1)], f"eng48 STAGES={s}")
for t in (7, 10, 15, 23):
    add(48, [("TILE_SIZE", t), ("LDGA", 0), ("TILEK", 2), ("STAGES", 0), ("LDGC", 1)], f"eng48 TILE_SIZE={t}")
add(46, [("TILE_SIZE", 4), ("KBLOCK", 2), ("STAGES", 3)], "eng46 基线(=heuristic 候选6)")
for kb in range(6):
    add(46, [("TILE_SIZE", 4), ("KBLOCK", kb), ("STAGES", 3)], f"eng46 KBLOCK={kb}")
for s in range(8):
    add(46, [("TILE_SIZE", 4), ("KBLOCK", 2), ("STAGES", s)], f"eng46 STAGES={s}")

PROFILE_TAGS = ("eng28 TILE_SIZE=", "eng46 KBLOCK=", "eng48 STAGES=", "eng3", "eng6")

out = open(os.path.join(HERE, "knob_sweep.txt"), "w")
kout = open(os.path.join(HERE, "knob_kernels.txt"), "w")
out.write(f"# 手工旋钮实测（fwd 图 4x3x64x64 -> 4x16x64x64, 20 iters/点）\n")
out.write(f"{'case':<34s} {'rc':>3s} {'plan_ms':>8s} {'ms/iter':>9s} {'maxerr':>10s}\n")
for eid, knobs, tag in CASES:
    rc, pm, ms, err, js = run(eid, knobs)
    kn = " ".join(f"{n}={v}" for n, v in knobs)
    out.write(f"{tag:<34s} {rc:>3d} {pm:>8.3f} {ms:>9.4f} {err:>10.2e}   [{kn}]\n")
    out.flush()
    print(f"{tag:<34s} rc={rc} plant={pm:.3f}ms {ms:.4f}ms/iter maxerr={err:.2e}")
    if rc == 0 and tag.startswith(PROFILE_TAGS):
        rc2, _, ms2, _, names = profile_kernels(eid, knobs)
        if not rc2:
            for nm in names:
                kout.write(f"{tag:<34s} {ms2:>9.4f}ms  {nm}\n")
            kout.flush()
out.close()
kout.close()
print("done ->", os.path.join(HERE, "knob_sweep.txt"))
