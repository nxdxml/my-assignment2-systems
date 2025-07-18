import time
import torch
from torch import nn
from contextlib import nullcontext


def test_mix_add():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

    """
看出加法低精度的误差很大
fp32 + fp32
fp16 + fp16
fp32 + fp16
fp32 + fp32转fp16
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
"""



class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1输出dtype-{x.dtype}")
        x = self.ln(x)
        print(f"ln输出dtype-{x.dtype}")
        x = self.fc2(x)
        print(f"fc2输出dtype-{x.dtype}")
        return x

# 运行一次前向和反向
def benchmarking_mixed_precision(model, x, use_mix, mix_type=torch.float32):
    device = x.device
    ctx = torch.autocast(device_type=device.type, dtype=mix_type) if use_mix else nullcontext()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    target = torch.randn_like(model(x), device=device)

    start_fwd = time.time()
    with ctx:
        y = model(x)
        loss = criterion(y, target)
        print(f"loss.dtype:{loss.dtype}")
    end_fwd = time.time()

    start_bwd = time.time()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"name{name}:dtype{param.dtype}")

    optimizer.step()
    end_bwd = time.time()
    print(f"mix_type:{mix_type}")
    print(f"fwd时间{end_fwd - start_fwd},use_mix:{use_mix}")
    print(f"bwd时间{end_bwd - start_bwd},use_mix:{use_mix}")
    print("=" * 50)




def main():
    # test_mix_add()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = [
        (1000, 500),
    ]

    for in_f, out_f in model_size:
        model = ToyModel(in_features=in_f, out_features=out_f).to(device)
        x = torch.randn((32, in_f), device=device)
        benchmarking_mixed_precision(
            model=model,
            x=x,
            use_mix=False,
        )

        benchmarking_mixed_precision(
            model=model,
            x=x,
            use_mix=True,
            mix_type=torch.float16,
        )

        benchmarking_mixed_precision(
            model=model,
            x=x,
            use_mix=True,
            mix_type=torch.bfloat16,
        )
        """
fc1输出dtype-torch.float32
ln输出dtype-torch.float32
fc2输出dtype-torch.float32
fc1输出dtype-torch.float32
ln输出dtype-torch.float32
fc2输出dtype-torch.float32
loss.dtype:torch.float32
namefc1.weight:dtypetorch.float32
nameln.weight:dtypetorch.float32
nameln.bias:dtypetorch.float32
namefc2.weight:dtypetorch.float32
mix_type:torch.float32
fwd时间0.00992131233215332,use_mix:False
bwd时间0.03929471969604492,use_mix:False
==================================================
fc1输出dtype-torch.float32
ln输出dtype-torch.float32
fc2输出dtype-torch.float32
fc1输出dtype-torch.float16
ln输出dtype-torch.float32
fc2输出dtype-torch.float16
loss.dtype:torch.float32
namefc1.weight:dtypetorch.float32
nameln.weight:dtypetorch.float32
nameln.bias:dtypetorch.float32
namefc2.weight:dtypetorch.float32
mix_type:torch.float16
fwd时间0.09343075752258301,use_mix:True
bwd时间0.009998798370361328,use_mix:True
==================================================
fc1输出dtype-torch.float32
ln输出dtype-torch.float32
fc2输出dtype-torch.float32
fc1输出dtype-torch.bfloat16
ln输出dtype-torch.float32
fc2输出dtype-torch.bfloat16
loss.dtype:torch.float32
namefc1.weight:dtypetorch.float32
nameln.weight:dtypetorch.float32
nameln.bias:dtypetorch.float32
namefc2.weight:dtypetorch.float32
mix_type:torch.bfloat16
fwd时间0.07174539566040039,use_mix:True
bwd时间0.0,use_mix:True
========================================
        """













if __name__ == "__main__":
    main()