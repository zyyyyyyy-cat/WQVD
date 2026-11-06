"""Initialization shared by the lightweight experimental branches."""
import torch.nn as nn


def init_branch(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.MultiheadAttention):
            nn.init.trunc_normal_(layer.in_proj_weight, std=0.02)
            if layer.in_proj_bias is not None:
                nn.init.zeros_(layer.in_proj_bias)
        elif isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def validate_dimensions(dim, num_query, num_heads):
    if dim <= 0 or num_query <= 0 or num_heads <= 0 or dim % num_heads:
        raise ValueError('Positive dimensions/queries/heads and DIM divisible by NUM_HEADS are required')
