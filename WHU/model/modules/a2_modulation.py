"""QVAM VAD/AFM adapted according to idea §3 (no prompt/CLS value bypass)."""
import torch
from torch import nn
from .common import init_branch, validate_dimensions


class A2FeatureModulation(nn.Module):
    def __init__(self, in_dim=768, dim=192, num_query=4, num_heads=4,
                 gamma=0.2, mask_out_init_std=0.005):
        super().__init__()
        validate_dimensions(dim, num_query, num_heads)
        if not 0 <= gamma < 1 or mask_out_init_std <= 0:
            raise ValueError('A2 requires 0 <= GAMMA < 1 and nonzero mask initialization')
        self.gamma = gamma
        self.patch_proj = nn.Linear(in_dim, dim)
        self.query_tokens = nn.Parameter(torch.empty(num_query, dim))
        self.vad_self_norm = nn.LayerNorm(dim)
        self.vad_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.vad_cross_norm = nn.LayerNorm(dim)
        self.vad_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # One lightweight FFN, only after patch reading (QVAM Eq. 1).
        self.vad_ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                     nn.GELU(), nn.Linear(2 * dim, dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        self.cls_proj = nn.Linear(in_dim, dim)
        self.afm_query_norm = nn.LayerNorm(dim)
        self.afm_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mask_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                       nn.GELU(), nn.LayerNorm(2 * dim),
                                       nn.Linear(2 * dim, in_dim))
        init_branch(self)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.mask_head[-1].weight, std=mask_out_init_std)
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(self, cls, patches, return_details=False):
        if patches.shape[1] == 0:
            raise ValueError('A2 needs at least one patch')
        patches = self.patch_proj(patches)
        queries = self.query_tokens.unsqueeze(0).expand(cls.shape[0], -1, -1)
        normed = self.vad_self_norm(queries)
        qs = queries + self.vad_self_attn(normed, normed, normed, need_weights=False)[0]
        cond = self.vad_cross_attn(self.vad_cross_norm(qs), patches, patches,
                                   need_weights=False)[0]
        cond = cond + self.vad_ffn(cond)  # No addition of qs.
        m0 = self.mask_token + self.cls_proj(cls.detach()).unsqueeze(1)
        read = self.afm_cross_attn(self.afm_query_norm(m0), cond, cond,
                                   need_weights=False)[0]  # No addition of m0.
        mask = torch.sigmoid(self.mask_head(read.squeeze(1)))
        z = cls * (1.0 + self.gamma * (2.0 * mask - 1.0))
        if return_details:
            return z, {'mask': mask, 'condition': cond}
        return z
