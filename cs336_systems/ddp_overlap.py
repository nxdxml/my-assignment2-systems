import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Set


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._allreduce_handles: List[Tuple[torch.Tensor, dist.Work]] = []
        self._registered_params: Set[torch.nn.Parameter] = set()

        # 广播参数（确保 rank 0 的模型权重被复制到所有进程）
        self._broadcast_model_parameters()

        # 为每个唯一的参数注册 post_accumulate_grad_hook
        for param in self.module.parameters():
            if param.requires_grad and param not in self._registered_params:
                param.register_post_accumulate_grad_hook(self._make_hook(param))
                self._registered_params.add(param)

    def _broadcast_model_parameters(self):
        """
        使用广播确保所有 rank 拥有相同的模型参数初始化。
        """
        for param in self.module.state_dict().values():
            if isinstance(param, torch.Tensor):
                dist.broadcast(param, src=0)

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(param: torch.nn.Parameter):
            # “通信句柄”（communication handle）在 PyTorch 分布式中是指一个对象（handle），它代表一个正在进行中的异步通信操作。
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._allreduce_handles.append((param.grad, handle))
        return hook

    def finish_gradient_synchronization(self):
        """
        等待所有异步 all_reduce 完成并进行平均。
        """
        world_size = dist.get_world_size()
        for grad, handle in self._allreduce_handles:
            handle.wait()
            grad /= world_size
        self._allreduce_handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # def parameters(self, recurse: bool = True):
    #     return self.module.parameters(recurse=recurse)

    # def named_parameters(self, prefix: str = '', recurse: bool = True):
    #     return self.module.named_parameters(prefix=prefix, recurse=recurse)
