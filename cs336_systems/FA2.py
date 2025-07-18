import torch
import math
from einops import rearrange, einsum
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
        raise NotImplementedError("Backward not implemented.")
