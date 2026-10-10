"""实测 cudnnCreate 返回的 handle 到底是什么：地址、开销、内部存的 stream、NULL 行为、线程相关性。

用法：/home/liquan/venvs/torch/bin/python probe_handle.py
"""
import ctypes
import struct
import threading
import time

import torch

NV = "/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia"
cudnn = ctypes.CDLL(f"{NV}/cudnn/lib/libcudnn.so.9")
libc = ctypes.CDLL("libc.so.6")

cudnn.cudnnGetVersion.restype = ctypes.c_size_t
cudnn.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudnn.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudnn.cudnnGetStream.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
for fn in (cudnn.cudnnCreate, cudnn.cudnnSetStream, cudnn.cudnnGetStream):
    fn.restype = ctypes.c_int
libc.malloc_usable_size.restype = ctypes.c_size_t
libc.malloc_usable_size.argtypes = [ctypes.c_void_p]

STATUS = {0: "SUCCESS", 1: "NOT_INITIALIZED", 2000: "BAD_PARAM", 2002: "NOT_SUPPORTED"}


def scan(handle, value):
    """在句柄对象所在内存页内查找某个指针值，返回偏移列表。"""
    page = 4096
    limit = page - (handle.value % page)
    buf = ctypes.string_at(handle, limit)
    return [hex(off) for off in range(0, limit - 8, 8)
            if struct.unpack_from("<Q", buf, off)[0] == value]


def main():
    torch.cuda.init()
    print(f"cuDNN {cudnn.cudnnGetVersion()}  设备: {torch.cuda.get_device_name(0)}\n")

    t0 = time.perf_counter()
    h1 = ctypes.c_void_p()
    print("首次 cudnnCreate ->", STATUS[cudnn.cudnnCreate(ctypes.byref(h1))], hex(h1.value),
          f"耗时 {(time.perf_counter() - t0) * 1e3:.2f} ms")
    t0 = time.perf_counter()
    h2 = ctypes.c_void_p()
    cudnn.cudnnCreate(ctypes.byref(h2))
    print("再次 cudnnCreate ->", hex(h2.value), f"耗时 {(time.perf_counter() - t0) * 1e3:.3f} ms",
          f"句柄堆块大小 {libc.malloc_usable_size(h2)} 字节（两次地址不同：{h1.value != h2.value}）")

    got = ctypes.c_void_p()
    cudnn.cudnnGetStream(h1, ctypes.byref(got))
    print(f"\n未设置 stream 时 cudnnGetStream -> {got.value}（NULL）")

    s1 = torch.cuda.Stream().cuda_stream
    s2 = torch.cuda.Stream().cuda_stream
    for name, s in (("s1", s1), ("s2", s2)):
        cudnn.cudnnSetStream(h1, s)
        cudnn.cudnnGetStream(h1, ctypes.byref(got))
        print(f"cudnnSetStream({name}) 后 getStream={hex(got.value)}，匹配={got.value == s}，"
              f"该指针出现在句柄内存偏移 {scan(h1, s)}")

    print(f"\ncudnnSetStream(NULL 句柄, 0) -> {cudnn.cudnnSetStream(None, 0)} "
          f"({STATUS.get(2000)})：句柄是几乎所有 API 的必需第一个参数")

    res = {}

    def worker():
        t = ctypes.c_void_p()
        cudnn.cudnnCreate(ctypes.byref(t))
        res["addr"] = t.value

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    print(f"\n另一个线程里 cudnnCreate 得到 {hex(res['addr'])}（与主线程 {hex(h1.value)} 不同）")


if __name__ == "__main__":
    main()
