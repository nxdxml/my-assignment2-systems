数据并行相关，single-node 实现ddp的一些总结

## 初始化相关和一些实验注意细节
每个gpu/cpu一个进程，初始化操作可以如下理解:
``` py
# 体现single-node，就是所有进程运行在本地
os.environ["MASTER_ADDR"] = "localhost"

# 相当于启动world_size个线程，每个线程运行fn函数，传入参数args
# join等待所有子进程执行完成后再继续主进程。
mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)

# fn第一个参数固定为rank，代表当前卡编号
fn(rank, *args) 

# 初始化进程组，采用gloo(适用cpu)后端进行通信,提供all-reduce之类的通信原语
# 主进程在rank=0上执行
dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

进行实验需要注意一些细节
1. 进行nccl通信的时候warmup=5个epoch
2. torch.cuda.synchronize() 进行等待
3. 在一台机器上进行实验

## 实验思路
DDP思想，将数据分成几份分到多张卡上，每个卡保存完整的模型。

这个思想在训练中就仅需要同步梯度，每个节点有各自数据方向传播的梯度，需要all-reduce-sum然后求平均，插入的地方如下
``` py
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()

    # 手动同步梯度：逐个参数 all_reduce 然后除以 world_size
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

    optimizer.step()
```

可以通过如下方法优化
1. 减小通信操作的数量，把参数合成一个大tensor通过一次通信完成，实测效果确实好一些
``` py
# 使用工具
torch._utils._flatten_dense_tensors 
torch._utils._unflatten_dense_tensors

# 测试结果
展开 平均通信时间  0.000642s
没展开 平均通信时间  0.001197s
测试epoch 100 前5轮warmup未计入统计时间
```
2. 将通信和计算重叠 (Overlapping)
梯度准备好可以立刻进行通信，不必等所有计算完毕。

这部分实践待补充

3. 上面同样准备好一部分数据进行通信，减小频繁的通信开销

这部分实践待补充

## 其余并行简单写写理解
+ 张量并行 -> 切分矩阵，适合节点内（通信量要求高），易于实现
+ 模型并行 -> 模型按层分到多个机器上(理解通信相比张量并行没那么高)
+ 流水线并行 -> 模型并行+ min_batch等操作减小气泡，会在系统上比较复杂
+ FSDP -> 理解是模型并行+deepspeed zero3(分优化器状态，模型参数，梯度)
+ EP -> MOE架构本来就很适合分到多张卡上，应该会有较为复杂的路由和系统实现
+ SP -> Megatron用了，在seplen维度并行，较少了激活值，感觉就是啥都得分这部分前面的方法没分到就加上了
弄个并行策略的方法->先搞TP->还是放不下考虑PP->把上面的并行模式用DP扩展

## 如果有空再整理一些参数的分析，包括总参数量，分成哪几部分，面试问的多的整理啥的