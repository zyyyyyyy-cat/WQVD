"""ViSA-inspired selection/relations; the explicit Lite adaptation in idea §2."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import init_branch, validate_dimensions


class A1LocalEnhancement(nn.Module):
    def __init__(self, in_dim=768, dim=192, num_query=2, topk=16,
                 num_heads=4, tau=0.1, gamma=0.05):
        super().__init__()
        validate_dimensions(dim, num_query, num_heads)
        if topk <= 0 or tau <= 0 or gamma < 0:
            raise ValueError('A1 requires TOPK/Tau > 0 and GAMMA >= 0')
        self.dim, self.num_query = dim, num_query
        self.topk, self.tau, self.gamma = topk, tau, gamma
        self.patch_norm = nn.LayerNorm(in_dim)
        self.patch_proj = nn.Linear(in_dim, dim)
        self.query_tokens = nn.Parameter(torch.empty(num_query, dim))
        self.query_proj = nn.Linear(in_dim, num_query * dim)
        self.relation_norm = nn.LayerNorm(dim)
        self.relation_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(num_query * dim, in_dim)
        self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        init_branch(self)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, cls, patches, return_details=False):
        b, n, _ = patches.shape
        if n == 0:
            raise ValueError('A1 needs at least one patch')
        values = self.patch_proj(self.patch_norm(patches))
        query = self.query_tokens.unsqueeze(0) + self.query_proj(cls.detach()).reshape(
            b, self.num_query, self.dim)
        # Explicitly disable autocast: float() alone does not keep matmul in FP32.
        with torch.autocast(device_type=values.device.type, enabled=False):
            scores = F.normalize(query.float(), dim=-1) @ F.normalize(
                values.float(), dim=-1).transpose(1, 2)
            top_scores, indices = scores.topk(min(self.topk, n), dim=-1)
            weights = F.softmax(top_scores / self.tau, dim=-1)
        k = indices.shape[-1]
        neighborhoods = values.unsqueeze(1).expand(-1, self.num_query, -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, self.dim))
        neighborhoods = neighborhoods.reshape(b * self.num_query, k, self.dim)
        normed = self.relation_norm(neighborhoods)
        related = neighborhoods + self.relation_attn(
            normed, normed, normed, need_weights=False)[0]
        related = related.reshape(b, self.num_query, k, self.dim)
        local = (weights.to(related.dtype).unsqueeze(-1) * related).sum(dim=2)
        u = self.output_norm(self.output_proj(local.reshape(b, -1)))
        z = cls + self.gamma * u
        if return_details:
            return z, u, {'top_scores': top_scores, 'weights': weights,
                          'indices': indices, 'query': query}
        return z, u
