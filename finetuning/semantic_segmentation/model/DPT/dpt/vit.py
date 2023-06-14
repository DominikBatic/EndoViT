import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F

# get the MAE model code
import sys
from pathlib import Path
sys.path.append("./pretraining/mae")
from prepare_mae_model_seg_version import prepare_mae_model


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

# NOTE: forward_vit and forward_flex are just a clever way to make any ViT backbone process any image shape, regardless of the image shape 
#       the backbone was pretrained on
def forward_vit(pretrained, x):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    if (not (gs_old == gs_h and gs_old == gs_w)):
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
    **kwargs
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    pretrained_dict = {"model": pretrained}

    if ("backbone_param_groups" in kwargs):
        pretrained_dict["param_groups"] = kwargs["backbone_param_groups"]

    return pretrained_dict


def _make_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
    **kwargs
):
    pretrained = nn.Module()

    pretrained.model = model

    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.attention = attention

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    if use_vit_only == True:
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
    else:
        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )

# ADDED
def _make_MAE_pretrained_vitb16_224(
    mae_hyperparams, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    # If an empty dictionary was passed, set up default hyperparameters.
    if (not mae_hyperparams):
        mae_hyperparams = {
            "mae_model"                     : "vit_base_patch16",
            "nb_classes"                    : 13, # doesn't matter what you set
            "drop_path"                     : 0.1,
            "pool_type"                     : "global_pool", # doesn't matter what you set
            "mae_ckpt"                      : "/home/guests/dominik_batic/thesis/final/results/pretraining/run_14_final_model_Seg/full_run/seed_1665/best_swa_ckpt_14_loss_0.1144.pth",
            "freeze_weights"                : -1,
            "reinit_n_layers"               : -1,
            "return_mae_optimizer_groups"   : True,
            "mae_weight_decay"              : 0.05,
            "mae_layer_decay"               : 0.65,
        }

    mae_dict = prepare_mae_model(
                mae_hyperparams["mae_model"], # default: vit_base_patch16 
                mae_hyperparams["nb_classes"], # default: 13 
                mae_hyperparams["drop_path"], # default: 0.1
                mae_hyperparams["pool_type"], # default: "global_pool" 
                mae_hyperparams["mae_ckpt"], # default: "/home/guests/dominik_batic/thesis/final/results/pretraining/run_14_final_model_Seg/full_run/seed_1665/best_swa_ckpt_14_loss_0.1144.pth",
                mae_hyperparams["freeze_weights"], # default: -1
                mae_hyperparams["reinit_n_layers"], # default: -1
                mae_hyperparams["return_mae_optimizer_groups"], # default: True
                mae_hyperparams["mae_weight_decay"], # default: 0.05 
                mae_hyperparams["mae_layer_decay"], # default 0.65
                verbose=False,
                debug=False
            )
    
    mae_model = mae_dict["model"]
    mae_param_groups = mae_dict["param_groups"]

    hooks = [2, 5, 8, 11] if hooks == None else hooks

    pretrained = _make_vit_b16_backbone(
        mae_model,
        features=[96, 192, 384, 768],
        size=[224, 224],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        backbone_param_groups = mae_param_groups,
    )

    return pretrained

# Image size needs to be changed
# solution: load a MAE pretrained model with prepare_mae_model and then pass it to _make_vit_b16_backbone, but with extra parameter
# size = [224, 224]
# careful: use_readout shoud be = "project"
def _make_pretrained_vitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
# END_ADDED


def _make_pretrained_deitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_deit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_distil_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model(
        "vit_deit_base_distilled_patch16_384", pretrained=pretrained
    )

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        start_index=2,
        enable_attention_hooks=enable_attention_hooks,
    )
