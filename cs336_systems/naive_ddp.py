import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import os
import time

device = "cuda"
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 50),
        )
    def forward(self, x):
        return self.net(x)


def setup_cpu(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)
    return torch.device("cpu")

def setup_gpu(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)
    return torch.device(f"cuda:{rank}")


def cleanup():
    dist.destroy_process_group()


def ddp_individual_parameters_on_after_backward(model, optimizer):
    # 手搓all-reduce平均梯度
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data)
            param.grad.data /= dist.get_world_size()

# 对比上面把所有参数合成了一个tensor，较少通信开销
def ddp_flattened_allreduce_on_after_backward(model, optimizer):
    # 收集所有参数梯度（非 None）
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    
    # Flatten 所有梯度为一个 tensor
    flat_grads = torch._utils._flatten_dense_tensors(grads)

    # All-reduce 平均化（通信）
    dist.all_reduce(flat_grads)
    flat_grads /= dist.get_world_size()

    # Unflatten 回原来的梯度形状
    synced_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)

    # 拷贝同步梯度回 param.grad
    for param, synced in zip(model.parameters(), synced_grads):
        if param.grad is not None:
            param.grad.copy_(synced)


def get_ddp_individual_parameters(model):
    # 同步初始参数，广播rank0权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def train_epoch(rank, world_size, base_data, base_labels, base_model, ddp_model, base_optimizer, ddp_optimizer, device):
    local_bs = base_data.size(0) // world_size
    loss_fn = nn.MSELoss()

    warpup_round = 5
    tol_round = 100
    comm_total_time = 0.0
    for epoch in range(tol_round):
        # 清空梯度
        base_optimizer.zero_grad()
        ddp_optimizer.zero_grad()

        # 开始
        total_start = time.time()

        # base_model用全部数据训练
        base_outputs = base_model(base_data.to(device))
        base_loss = loss_fn(base_outputs, base_labels.to(device))
        base_loss.backward()
        base_optimizer.step()

        # ddp_model每个rank只用对应切片
        start = rank * local_bs
        end = start + local_bs
        ddp_inputs = base_data[start:end].to(device)
        ddp_targets = base_labels[start:end].to(device)

        ddp_outputs = ddp_model(ddp_inputs)
        ddp_loss = loss_fn(ddp_outputs, ddp_targets)
        ddp_loss.backward()

        # 手搓同步梯度
        comm_start = time.time()
        ddp_individual_parameters_on_after_backward(ddp_model, ddp_optimizer)
        # ddp_flattened_allreduce_on_after_backward(ddp_model, ddp_optimizer)
        comm_end = time.time()

        if epoch >= warpup_round:
            comm_total_time += (comm_end - comm_start)

        ddp_optimizer.step()

        total_end = time.time()

        
        # rank0验证权重相等，通过暂时关闭
        # if rank == 0:
        #     for bp, dp in zip(base_model.parameters(), ddp_model.parameters()):
        #         assert torch.allclose(bp, dp, atol=1e-6), f"参数不匹配，epoch={epoch}"
        #     print(f"Epoch {epoch} 参数一致验证通过")

        # shuffle数据（rank0执行，其他rank同步），修改方案为下面的perm
        # if rank == 0:
        #     perm = torch.randperm(base_data.size(0))
        #     base_data[:] = base_data[perm]
        #     base_labels[:] = base_labels[perm]
        # dist.broadcast(base_data, src=0)
        # dist.broadcast(base_labels, src=0)

        if rank == 0:
            perm = torch.randperm(base_data.size(0), device=device)
        else:
            perm = torch.empty(base_data.size(0), dtype=torch.long, device=device)

        dist.broadcast(perm, src=0)

        base_data = base_data[perm]
        base_labels = base_labels[perm]

        if rank == 0:
            print(f"[Epoch {epoch}] 总时间: {total_end - total_start:.6f}s, 通信时间: {comm_end - comm_start:.6f}s")
    if rank == 0:
        avg_comm_time = comm_total_time / (tol_round - warpup_round)
        print(f"平均通信时间  {avg_comm_time:.6f}s")

def train_ddp(rank, world_size, base_data, base_labels):
    # device = setup_cpu(rank, world_size)
    device = setup_gpu(rank, world_size)

    base_data = base_data.to(device)
    base_labels = base_labels.to(device)

    base_model = ToyModel().to(device)
    ddp_model = deepcopy(base_model)
    ddp_model = get_ddp_individual_parameters(ddp_model)
    base_optimizer = optim.SGD(base_model.parameters(), lr=0.1)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)

    train_epoch(rank, world_size, base_data, base_labels, base_model, ddp_model, base_optimizer, ddp_optimizer, device)

    cleanup()


if __name__ == "__main__":
    world_size = 2
    batch_size = 20
    input_dim = 100
    output_dim = 50

    torch.manual_seed(0)

    base_data = torch.randn(batch_size, input_dim, device=device)
    base_labels = torch.randn(batch_size, output_dim, device=device)



    mp.spawn(
        train_ddp,
        args=(world_size, base_data, base_labels),
        nprocs=world_size,
        join=True,
    )
