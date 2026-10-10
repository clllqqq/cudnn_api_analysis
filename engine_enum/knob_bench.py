#!/usr/bin/env python3
"""对若干"engine+旋钮"点做稳定计时（多次重复取最好值）。输出 knob_bench.txt。"""
import ctypes, os
lib = ctypes.CDLL("/home/liquan/cudnn_api_analysis/engine_enum/librunner.so", mode=ctypes.RTLD_GLOBAL)
lib.run_engine_global_knobs.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.run_engine_global_knobs.restype = ctypes.c_int
K = dict(SPLIT_K=0, TILE_SIZE=2, USE_TEX=3, KBLOCK=5, LDGA=6, MULTIPLY=11, SPLIT_K_BUF=12,
         TILEK=13, STAGES=14, REDUCTION_MODE=15, SPLIT_K_SLC=17, IDX_MODE=18, LDGC=22,
         SPECFILT=23, KERNEL_CFG=24, WORKSPACE=25, STREAM_K=38)

def once(eid, knobs, iters):
    types = (ctypes.c_int * len(knobs))(*[K[n] for n, _ in knobs])
    vals = (ctypes.c_longlong * len(knobs))(*[v for _, v in knobs])
    pm, ms, err = (ctypes.c_double(0) for _ in range(3))
    rc = lib.run_engine_global_knobs(eid, types, vals, len(knobs), iters, None, 0,
                                     ctypes.byref(pm), ctypes.byref(ms), ctypes.byref(err))
    return rc, ms.value, err.value

def bench(eid, knobs, iters=200, reps=3):
    best, err, rc0 = None, None, 0
    for _ in range(reps):
        rc, ms, e = once(eid, knobs, iters)
        if rc: return rc, 0.0, 0.0
        best = ms if best is None or ms < best else best
        err = e
    return 0, best, err

CASES = []
for t in range(5):
    CASES.append((28, [("TILE_SIZE", t), ("USE_TEX", 0)], 200, f"eng28 TILE_SIZE={t}"))
CASES.append((28, [("TILE_SIZE", 4), ("USE_TEX", 1)], 200, "eng28 TILE_SIZE=4 USE_TEX=1"))
for t in range(5):
    CASES.append((1, [("TILE_SIZE", t), ("USE_TEX", 0)], 200, f"eng1 TILE_SIZE={t}"))
for w in range(3):
    CASES.append((6, [("WORKSPACE", w)], 200, f"eng6(Winograd) WORKSPACE={w}"))
for m in range(3):
    CASES.append((3, [("MULTIPLY", m)], 20, f"eng3(FFT) MULTIPLY={m}"))
for tag, kn in [("cand6  {4,2,3}", [("TILE_SIZE",4),("KBLOCK",2),("STAGES",3)]),
                ("cand9  {1,3,2}", [("TILE_SIZE",1),("KBLOCK",3),("STAGES",2)]),
                ("cand10 {5,2,6}", [("TILE_SIZE",5),("KBLOCK",2),("STAGES",6)]),
                ("cand13 {5,3,4}", [("TILE_SIZE",5),("KBLOCK",3),("STAGES",4)])]:
    CASES.append((46, kn, 200, f"eng46 {tag}"))
for tag, kn in [("cand2 {2,2,0,1}", [("TILE_SIZE",2),("LDGA",0),("TILEK",2),("STAGES",0),("LDGC",1)]),
                ("cand3 {15,0,0,0}", [("TILE_SIZE",15),("LDGA",0),("TILEK",0),("STAGES",0),("LDGC",0)])]:
    CASES.append((48, kn, 200, f"eng48 {tag}"))

lines = ["# engine + 旋钮 稳定计时（N=4 C=3 64x64 -> K=16；reps=3 取最小值）",
         f"{'case':<34s} {'iters':>5s} {'ms/iter':>9s} {'maxerr':>10s}"]
for eid, knobs, iters, tag in CASES:
    rc, ms, err = bench(eid, knobs, iters)
    lines.append(f"{tag:<34s} {iters:>5d} {ms:>9.4f} {err:>10.2e}" + ("" if rc == 0 else f"  rc={rc}"))
    print(lines[-1])
open("knob_bench.txt", "w").write("\n".join(lines) + "\n")
