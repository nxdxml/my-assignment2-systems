import torch
import math
from einops import rearrange, einsum
import triton
import triton.language as tl

TILE_SIZE = 16

class FA2_Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        # tile 成 16 x 16的block
        B_q = 16
        B_k = 16
        B, N_q, d = Q.shape # Q 大小
        _, N_k, _ = K.shape # K,V 大小

        O = torch.empty(B, N_q, d)
        L = torch.empty(B, N_q)
        
        T_q = (N_q + B_q - 1) // B_q # Q 分成多少块[B_q, d]
        T_k = (N_k + B_k - 1) // B_k # K,V 分成多少块[B_k, d]

        d_sqrt = math.sqrt(d)
        
        for b in range(B):
            for i in range(T_q):
                Q_i = Q[b, i*B_q : (i+1)*B_q, :] # [B_q, d]
                # 我理解为上一个
                O_i_0 = torch.zeros(B_q, d)
                l_i_0 = torch.zeros(B_q)
                m_i_0 = torch.full((B_q,), float("-inf"))

                for j in range(T_k):
                    K_j = K[b, j*B_k : (j+1)*B_k, :]
                    V_j = V[b, j*B_k : (j+1)*B_k, :]
                    S_i_j = einsum(Q_i, K_j, "B_q d, B_k d -> B_q B_k") / d_sqrt
                    m_i_j = torch.maximum(m_i_0, S_i_j.max(dim=-1).values) # [B_q]
                    P_i_j = torch.exp(S_i_j - m_i_j.unsqueeze(-1)) # [B_q, B_k]
                    l_i_j = torch.exp(m_i_0 - m_i_j) * l_i_0 + P_i_j.sum(dim=-1) # [B_q]
                    # [B_q, 1] * [B_q, d] 广播机制
                    O_i_j = torch.exp(m_i_0 - m_i_j).unsqueeze(-1) * O_i_0 + P_i_j @ V_j # [B_q, d]
                    # 存储上一项
                    O_i_0 = O_i_j
                    l_i_0 = l_i_j
                    m_i_0 = m_i_j
                O_i = O_i_0 / l_i_0.unsqueeze(-1)
                l_i = m_i_0 + torch.log(l_i_0) # [B_q]

                O[b, i*B_q : (i+1)*B_q, :] = O_i
                L[b, i*B_q : (i+1)*B_q] = l_i
        # 保存必须的张量，用于方向计算
        ctx.save_for_backward(Q, K, V, O, L)
        # ctx.is_causal = is_causal
        return O

    # 可以不用实现
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V,O, L = ctx.saved_tensors  # L 是 logsumexp 值 [B, N_q]
        B, N_q, d = Q.shape
        _, N_k, _ = K.shape
        d_sqrt = math.sqrt(d)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for b in range(B):
            Q_b = Q[b]  # [N_q, d]
            K_b = K[b]  # [N_k, d]
            V_b = V[b]  # [N_k, d]
            O_b = O[b]
            L_b = L[b]  # [N_q]
            dO_b = grad_output[b]  # [N_q, d]

            # === (13) S = QK^T / sqrt(d)
            S = Q_b @ K_b.T / d_sqrt  # [N_q, N_k]

            # === (14) P_ij = exp(S_ij - L_i) = attention probs (not normalized)
            # L is logsumexp, so exp(S - L[:, None]) gives P
            P = torch.exp(S - L_b[:, None])  # [N_q, N_k]

            # === (15) dV = P^T @ dO
            dV[b] = P.T @ dO_b  # [N_k, d]

            # === (16) dP = dO @ V^T
            dP = dO_b @ V_b.T  # [N_q, N_k]

            # === (17) dS = P * (dP - Di)
            Di = (O_b * dO_b).sum(dim=-1, keepdim=True) 

            dS = P * (dP - Di)  # [N_q, N_k]

            # === (18) dQ = dS @ K / sqrt(d)
            dQ[b] = dS @ K_b / d_sqrt

            # === (19) dK = dS^T @ Q / sqrt(d)
            dK[b] = dS.T @ Q_b / d_sqrt

        return dQ, dK, dV, None  # is_causal has no gradient


class FA2_Tririon(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        device = Q.device
        dtype = Q.dtype
        # tile 成 16 x 16的block
        B_q = 16
        B_k = 16
        B, N_q, d = Q.shape # Q 大小
        _, N_k, _ = K.shape # K,V 大小

        O = torch.empty((B, N_q, d), device=device, dtype=dtype)
        L = torch.empty((B, N_q), device=device, dtype=dtype)
        
        # T_q = (N_q + B_q - 1) // B_q # Q 分成多少块[B_q, d]
        # T_k = (N_k + B_k - 1) // B_k # K,V 分成多少块[B_k, d]

        scale = 1.0 / math.sqrt(d)


        flash_fwd_kernel[(N_q // B_q, B)](
            Q, K, V, # 输入QKV指针
            O, L, # 输出指针
            Q.stride(0), Q.stride(1), Q.stride(2), # Q[B, N_q, d]
            K.stride(0), K.stride(1), K.stride(2), # K[B, N_k, d]
            V.stride(0), V.stride(1), V.stride(2), # V[B, N_k, d]
            O.stride(0), O.stride(1), O.stride(2), # O[B, N_q, d]
            L.stride(0), L.stride(1), # L[B, N_q]
            N_q, N_k, # 
            scale, # 应该是 sqrt(D)^{-1}
            d, # d_model
            B_q, # Q维度分块大小
            B_k, # K维度分块大小
            is_causal, 
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward not implemented.")
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, # 输入QKV指针
    O_ptr, L_ptr, # 输出指针
    stride_qb, stride_qq, stride_qd, # Q[B, N_q, d]
    stride_kb, stride_kk, stride_kd, # K[B, N_k, d]
    stride_vb, stride_vk, stride_vd, # V[B, N_k, d]
    stride_ob, stride_oq, stride_od, # O[B, N_q, d]
    stride_lb, stride_lq, # L[B, N_q]
    N_QUERIES, N_KEYS, # 
    scale, # 应该是 sqrt(D)^{-1}
    D: tl.constexpr, # d_model
    Q_TILE_SIZE: tl.constexpr, # Q维度分块大小
    K_TILE_SIZE: tl.constexpr, # K维度分块大小
    is_casual : tl.constexpr, # 
):
    # 每个block自动不同
    query_tile_index = tl.program_id(0) # blockIdx.x
    batch_index = tl.program_id(1) # blockIdx.y

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, # Q的起始点
        shape=(N_QUERIES, D), # 处理数据的大小
        strides=(stride_qq, stride_qd), # 两个维度跳跃stride
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 取Q的起始地址
        block_shape=(Q_TILE_SIZE, D), # tile大小
        order=(1, 0), # 先列后行访问
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, # K的起始点
        shape=(N_KEYS, D), # 处理数据的大小
        strides=(stride_kk, stride_kd), # 两个维度跳跃stride
        offsets=(0, 0), # 取K的起始地址
        block_shape=(K_TILE_SIZE, D), # tile大小
        order=(1, 0), # 先列后行访问
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, # V的起始点
        shape=(N_KEYS, D), # 处理数据的大小
        strides=(stride_vk, stride_vd), # 两个维度跳跃stride
        offsets=(0, 0), # 取V的起始地址
        block_shape=(K_TILE_SIZE, D), # tile大小
        order=(1, 0), # 先列后行访问
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob, # O的起始点
        shape=(N_QUERIES, D), # 处理数据的大小
        strides=(stride_oq, stride_od), # 两个维度跳跃stride
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 取O的起始地址
        block_shape=(Q_TILE_SIZE, D), # tile大小
        order=(1, 0), # 先列后行访问
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, # L的起始点
        shape=(N_QUERIES,), # 处理数据的大小
        strides=(stride_lq,), # 维度跳跃stride
        offsets=(query_tile_index * Q_TILE_SIZE,), # 取L的起始地址
        block_shape=(Q_TILE_SIZE,), # tile大小
        order=(0,), # 先列后行访问
    )

    Q_i = tl.load(Q_block_ptr)

    O_i_0 = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32) # [B_q, d]
    l_i_0 = tl.zeros((Q_TILE_SIZE, ), dtype = tl.float32) # [B_q]
    m_i_0 = tl.full((Q_TILE_SIZE, ), float("-inf"), dtype = tl.float32) # [B_q]

    # d_sqrt = math.sqrt(D)

    # 移到前面避免反复计算
    if is_casual:
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # [B_q]

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # 加载数据
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        # S_i_j = einsum(Q_i, K_j, "B_q d, B_k d -> B_q B_k") / d_sqrt
        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale

        # 加入mask
        if is_casual:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # [B_k]
            mask = k_idx[None, :] <= q_idx[:, None] # [B_q, B_k] k在q后为false
            S_i_j = tl.where(mask, S_i_j, float("-inf")) # 屏蔽false

        # m_i_j = torch.maximum(m_i_0, S_i_j.max(dim=-1).values)
        m_i_j = tl.maximum(m_i_0, tl.max(S_i_j, axis=1))

        # P_i_j = torch.exp(S_i_j - m_i_j.unsqueeze(-1))
        P_i_j = tl.exp(S_i_j - m_i_j[:, None])

        # l_i_j = torch.exp(m_i_0 - m_i_j) * l_i_0 + P_i_j.sum(dim=-1)
        l_i_j = tl.exp(m_i_0 - m_i_j) * l_i_0 + tl.sum(P_i_j, axis=1)

        # O_i_j = torch.exp(m_i_0 - m_i_j).unsqueeze(-1) * O_i_0 + P_i_j @ V_j
        O_i_j = tl.exp(m_i_0 - m_i_j)[:, None] * O_i_0 + tl.dot(P_i_j, V_j)

        O_i_0 = O_i_j
        l_i_0 = l_i_j
        m_i_0 = m_i_j

        # 移动指针,原地操作这里作业pdf例子可能有问题
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # O_i = O_i_0 / l_i_0.unsqueeze(-1)
    O_i = O_i_0 / l_i_0[:, None]
    L_i = m_i_0 + tl.log(l_i_0)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)




