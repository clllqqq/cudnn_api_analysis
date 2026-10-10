import os, torch, torch.nn as nn
print("V8_API_DISABLED =", os.environ.get("TORCH_CUDNN_V8_API_DISABLED"))
x = torch.randn(4,3,64,64,device="cuda", requires_grad=True)
nn.Conv2d(3,16,3,padding=1).cuda()(x).sum().backward()
torch.cuda.synchronize(); print("conv ok")
