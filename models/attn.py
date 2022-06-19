import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

#masked attn
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape #32,8,48,64
        _, _, L_Q, _ = Q.shape #32,8,48,64

        # calculate the sampled Q_K
        #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html, Returns a new tensor with a dimension of size one inserted at the specified position
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) #[32, 8, 1, 48, 64], #[32, 8, 48, 48, 64]
        ##https://pytorch.org/docs/stable/generated/torch.randint.html, Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # tensor of dimension(48,20) is cretaed filled with random values between 0 and 48
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] #[32, 8, 48, 20, 64]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze() #matmul([32, 8, 48, 1, 64],[32, 8, 48, 64, 20])
        #[32, 8, 48, 20]
        #https://pytorch.org/docs/stable/generated/torch.max.html
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) #[32,8,48]
        #https://pytorch.org/docs/stable/generated/torch.topk.html
        #Returns the k largest elements of the given input tensor along a given dimension.
        M_top = M.topk(n_top, sorted=False)[1] #[32,48,20]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q) [32, 8, 20, 64]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape #[32,8,48,64]
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            #https://pytorch.org/docs/stable/generated/torch.cumsum.html
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape #32,48,8,64

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            #https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape #32,48,8,8
        _, L_K, _, _ = keys.shape #48

        queries = queries.transpose(2,1) #32,8,48,64
        keys = keys.transpose(2,1) #32,8,48,64
        values = values.transpose(2,1) #32,8,48,64

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # 20
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # 20

        U_part = U_part if U_part<L_K else L_K #20
        u = u if u<L_Q else L_Q #20
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads) #64
        d_values = d_values or (d_model//n_heads) #64
        #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) 
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape #32,48
        _, S, _ = keys.shape #48
        H = self.n_heads #8

        queries = self.query_projection(queries).view(B, L, H, -1) #32,48,8,64
        keys = self.key_projection(keys).view(B, S, H, -1) #32,48,8,64
        values = self.value_projection(values).view(B, S, H, -1) #32,48,8,64

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
