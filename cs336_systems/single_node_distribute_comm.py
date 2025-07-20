import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    # 第一个进程所在的主机和端口，单机为localhost
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo",  # cpu 进程通信 backend
                            rank=rank, # 当前进程编号
                            world_size=world_size, # 进程总数
                            )

def dist_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"before all-reduce rank:{rank} data:{data}")
    dist.all_reduce(data, async_op=False)
    print(f"after all-reduce rank:{rank} data:{data}")




if __name__ == "__main__":
    world_size = 4 # 总的设备数量
    mp.spawn(fn=dist_demo, args=(world_size,), nprocs=world_size, join=True)