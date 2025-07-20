import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import itertools

import time


def setup(rank, world_size, backend):
    # 第一个进程所在的主机和端口，单机为localhost
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend,  # "gloo" or "cuda"
                            rank=rank, # 当前进程编号
                            world_size=world_size, # 进程总数
                            )

def benchmarking_all_reduce(
    rank,
    world_size,
    tensor_range_MB,
    backend,
    device,
    warmup=5,
):
    setup(rank=rank, world_size=world_size, backend=backend)

    x_len = tensor_range_MB * 1024 * 1024 // 4
    x = torch.randn(x_len, dtype=torch.float32, device=device)
    for _ in range(warmup):
        dist.all_reduce(x, async_op=False)
    
    # print(f"before all reduce rank:{rank},data_pre {x[0:3]}")
    
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    dist.all_reduce(x, async_op=False)
    
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    # print(f"after all reduce rank:{rank},data_pre {x[0:3]}")
    print(f"rank {rank} -> time {end_time - start_time}")
    dist.destroy_process_group() # nccl 没这个会报错


def run_benchmark(
    world_size,
    tensor_range_MB,
    backend,
    device,
    warmup=5,
):
    mp.spawn(fn=benchmarking_all_reduce,
              args=(world_size,tensor_range_MB,backend,device,warmup),
              nprocs=world_size,
              join=True,
              )

def main():
    # backends = ["gloo", "nccl"]
    backends = ["gloo"]
    devices = ["cpu", "cuda"]
    world_sizes = [2, 4, 6]
    tensor_sizes = [1, 10, 100, 1000]



    for backend, device in itertools.product(backends, devices):
        # Skip invalid combos (NCCL needs CUDA)
        if backend == "nccl" and device != "cuda":
            continue

        for world_size in world_sizes:
            for tensor_size in tensor_sizes:
                print("=" * 60)
                print(f"Running: backend={backend}, device={device}, "
                      f"world_size={world_size}, tensor={tensor_size}MB")
                run_benchmark(
                    world_size=world_size,
                    backend=backend,
                    tensor_range_MB=tensor_size,
                    device=device,
                )
                print("=" * 60)

if __name__ == "__main__":
    main()