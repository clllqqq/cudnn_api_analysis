"""legacy 卷积路径 + torch.backends.cudnn.benchmark=True：
证明 legacy API 也能"选 engine"——只是用 cudnnFindConvolution*AlgorithmEx 把 8 个 legacy 算法实测一遍再挑。

用法：
  TORCH_CUDNN_V8_API_DISABLED=1 CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=logs/fe_conv_legacy_bench.log \
    /home/liquan/venvs/torch/bin/python probe_legacy_bench.py
"""
import os
import torch
import torch.nn as nn

print("V8_API_DISABLED =", os.environ.get("TORCH_CUDNN_V8_API_DISABLED"))
torch.backends.cudnn.benchmark = True
print("cudnn.benchmark =", torch.backends.cudnn.benchmark)

x = torch.randn(4, 3, 64, 64, device="cuda", requires_grad=True)
nn.Conv2d(3, 16, 3, padding=1).cuda()(x).sum().backward()
torch.cuda.synchronize()
print("conv ok")
