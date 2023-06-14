# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, verbose=True):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    # ADDED
    # since we are applying LLRD to pre-training, we need to apply it to the decoder as well
    num_layers_decoder = len(model.decoder_blocks) + 1 if hasattr(model, "decoder_blocks") else 0
    # END_ADDED

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers, num_layers_decoder)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # WAS_BEFORE:
        #print("MAE parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    # ADDED 27.01.2023.
    if (verbose):
        separator_line = ''.join(['*']*200)
        print(f"\n\n{separator_line}\n{separator_line}\n")
        print("MAE parameter groups:")

        for g, (k, _) in enumerate(param_group_names.items()):
            print(f"{g:2d} -- {{'lr_scale': {param_group_names[k]['lr_scale']:.5f}, 'weight_decay': {param_group_names[k]['weight_decay']:.5f}, 'params': {param_group_names[k]['params']}}}")

        print(f"\n{separator_line}\n{separator_line}\n\n")

    # END_ADDED 27.01.2023.

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers, num_layers_decoder):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    # CHANGED:
    # since we are applying LLRD to pre-training, we need to apply it to the decoder as well
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    # ADDED
    elif name.startswith('decoder_norm') or name.startswith('decoder_pred'):
        return num_layers - num_layers_decoder
    elif name.startswith('decoder_blocks'):
        return num_layers_decoder - (int(name.split('.')[1]) + 1) + (num_layers - num_layers_decoder)
    # END_ADDED
    else:
        return num_layers