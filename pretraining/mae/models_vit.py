# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, pool_type='global_pool', **kwargs): #'cls_token'
        super(VisionTransformer, self).__init__(**kwargs)

        self.reinit_n_layers = -1

        self.pool_type = pool_type
        if self.pool_type == 'global_pool' or self.pool_type == 'no_pooling':
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

            # ADDED: init the normalization layer
            nn.init.constant_(self.fc_norm.bias, 0)
            nn.init.constant_(self.fc_norm.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.pool_type == 'global_pool':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.pool_type == 'no_pooling':
            x = x[:, 1:, :] # return per-patch embeddings
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0] # return only the cls token

        return outcome
    
    # ADDED: Re-init weights
    def reinit_weights(self, reinit_n_layers):
        self.reinit_n_layers = reinit_n_layers
        # Re-init last normalization layer from encoder
        nn.init.constant_(self.fc_norm.bias if self.pool_type == 'global_pool' or self.pool_type == 'no_pooling' else self.norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight if self.pool_type == 'global_pool' or self.pool_type == 'no_pooling' else self.norm.weight, 1.0)

        if (not isinstance(self.head, torch.nn.Identity)):
            trunc_normal_(self.head.weight, std=2e-5)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

        for i in range(reinit_n_layers):
            self.blocks[-(i+1)].apply(self._init_weights)

    def reinit_possible(self, reinit_n_layers):
        num_layers_encoder = len(self.blocks)

        if (reinit_n_layers > 0 and reinit_n_layers <= num_layers_encoder):
            return True
        else:
            return False
    
    # ADDED: checking if re-init works
    def _debug_reinit(self, text):
        print("".join(["*"] * 200))
        print("".join(["*"] * 200))

        print(f"\n{text}\nHead:\n", self.head.weight.data)
        num_blocks = len(self.blocks)

        for i, layer in enumerate(self.blocks[-self.reinit_n_layers:]):
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    print(f"\n{num_blocks - self.reinit_n_layers + i} nn.Linear.weight:\n", module.weight.data)
                    print(f"\n{num_blocks - self.reinit_n_layers + i} nn.Linear.bias:\n", module.bias.data)
                elif isinstance(module, nn.LayerNorm):
                    print(f"\n{num_blocks - self.reinit_n_layers + i} nn.LayerNorm.weight:\n", module.weight.data)
                    print(f"\n{num_blocks - self.reinit_n_layers + i} nn.LayerNorm.bias:\n", module.bias.data)
                    
        print("".join(["*"] * 200))
        print("".join(["*"] * 200))

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model