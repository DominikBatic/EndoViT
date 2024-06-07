#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Nwoye Chinedu Innocent
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as basemodels
import torchvision.transforms as transforms

import sys
sys.path.append("./pretraining/mae")
from prepare_mae_model import prepare_mae_model


OUT_HEIGHT = 8
OUT_WIDTH  = 14


#%% Model Rendezvous
class Rendezvous(nn.Module):
    """
    Rendezvous: attention mechanism for surgical action triplet recognition by Nwoye, C.I. et.al. 2021
    @args:
        image_shape: a tuple (height, width) e.g: (224,224)
        basename: Feature extraction network: e.g: "resnet50", "VGG19"
        num_tool: default = 6, 
        num_verb: default = 10, 
        num_target: default = 15, 
        num_triplet: default = 100, 
        layer_size: multi-head layers default = 8, 
        num_heads: default = 4, 
        d_model: mh-SCAM's feature depth. default = 128
    @call:
        inputs: Batch of input images of shape [batch, height, width, channel]
    @output: 
        enc_i: tuple (cam, logits) for instrument
        enc_v: tuple (cam, logits) for verb
        enc_t: tuple (cam, logits) for target
        dec_ivt: logits for triplet
    """
    # ADDED: **kwargs (for MAE hyperparameters)
    def __init__(self, basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, hr_output=False, use_ln=False, **kwargs):
        super(Rendezvous, self).__init__()
        # ADDED: **kwargs (for MAE hyperparameters)
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output, **kwargs)
        self.decoder = Decoder(layer_size, d_model, num_heads, num_triplet, use_ln=use_ln)    
     
    def forward(self, inputs):
        enc_i, enc_v, enc_t, enc_ivt = self.encoder(inputs)
        dec_ivt = self.decoder(enc_i, enc_v, enc_t, enc_ivt)
        return enc_i, enc_v, enc_t, dec_ivt
    

#%% Triplet Components Feature Encoder
class Encoder(nn.Module):
    # ADDED: **kwargs (for MAE hyperparameters)
    def __init__(self, basename='resnet18', num_tool=6,  num_verb=10, num_target=15, num_triplet=100, hr_output=False, **kwargs):
        super(Encoder, self).__init__()
        depth = 64 if basename == 'resnet18' else 128
        # ADDED: **kwargs (for MAE hyperparameters)
        self.basemodel  = BaseModel(basename, hr_output, **kwargs)
        self.wsl        = WSL(num_tool, depth)
        self.cagam      = CAGAM(num_tool, num_verb, num_target)
        self.bottleneck = Bottleneck(num_triplet)
        
    def forward(self, x):
        high_x, low_x = self.basemodel(x)
        enc_i         = self.wsl(high_x)
        enc_v, enc_t  = self.cagam(high_x, enc_i[0])
        enc_ivt       = self.bottleneck(low_x)
        return enc_i, enc_v, enc_t, enc_ivt


#%% MultiHead Attention Decoder
class Decoder(nn.Module):
    def __init__(self, layer_size, d_model, num_heads, num_class=100, use_ln=False):
        super(Decoder, self).__init__()        
        self.projection = nn.ModuleList([Projection(num_triplet=num_class, out_depth=d_model) for i in range(layer_size)])
        self.mhma       = nn.ModuleList([MHMA(num_class=num_class, depth=d_model, num_heads=num_heads, use_ln=use_ln) for i in range(layer_size)])
        self.ffnet      = nn.ModuleList([FFN(k=layer_size-i-1, num_class=num_class, use_ln=use_ln) for i in range(layer_size)])
        self.classifier = Classifier(num_class)
        
    def forward(self, enc_i, enc_v, enc_t, enc_ivt):
        X = enc_ivt.clone()
        for P, M, F in zip(self.projection, self.mhma, self.ffnet):
            X = P(enc_i[0], enc_v[0], enc_t[0], X)
            X = M(X)
            X = F(X)
        logits = self.classifier(X)
        return logits


#%% Feature extraction backbone
class BaseModel(nn.Module):
    # ADDED: **kwargs (for MAE hyperparameters)
    def __init__(self, basename='resnet18', hr_output=False, *args, **kwargs):
        super(BaseModel, self).__init__(*args)
        self.basename = basename

        self.output_feature = {} 
        if basename == 'resnet18':
            self.basemodel      = basemodels.resnet18(pretrained=True)     
            if hr_output: self.increase_resolution()
            self.basemodel.layer1[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))        
        if basename == 'resnet50':
            self.basemodel      = basemodels.resnet50(pretrained=True)
            self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
        # ADDED: ViT backbone
        if basename == 'mae':
            assert kwargs, "Missing MAE model hyperparameters!"
            self.basemodel      = prepare_mae_model(
                                        kwargs["mae_model"], # default: vit_base_patch16
                                        kwargs["nb_classes"], # default: 0, NOTE: should be kept 0, otherwise code likely crashes
                                        kwargs["drop_path"], # default: 0.1
                                        'no_pooling', 
                                        kwargs["mae_ckpt"], # default: ""
                                        kwargs["freeze_weights"] # default: 3
                                        )
            
            # TODO: allow different patch sizes, currently only supports patch_size=16 which results in 14x14 patches
            global OUT_HEIGHT, OUT_WIDTH
            OUT_HEIGHT = 14
            OUT_DITH = 14
            print("MAE uses different resolution output ({}x{})".format(OUT_HEIGHT,OUT_WIDTH))

            self.basemodel.blocks[3].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.blocks[11].bn2.register_forward_hook(self.get_activation('high_level_feature'))

            self.high_level_feature = nn.Conv2d(in_channels=self.basemodel.num_features, out_channels=512, kernel_size=1)
            self.low_level_feature = nn.Conv2d(in_channels=self.basemodel.num_features, out_channels=64, kernel_size=1)
        # END_ADDED
        
    def increase_resolution(self):  
        global OUT_HEIGHT, OUT_WIDTH  
        self.basemodel.layer3[0].conv1.stride = (1,1)
        self.basemodel.layer3[0].downsample[0].stride=(1,1)  
        self.basemodel.layer4[0].conv1.stride = (1,1)
        self.basemodel.layer4[0].downsample[0].stride=(1,1)
        OUT_HEIGHT *= 4
        OUT_WIDTH  *= 4
        print("using high resolution output ({}x{})".format(OUT_HEIGHT,OUT_WIDTH))        

    def get_activation(self, layer_name):
        def hook(module, input, output):
            self.output_feature[layer_name] = output
        return hook
    
    def forward(self, x):
        # ADDED
        output = self.basemodel(x) # [B, N, E] = [B, 196, 768] where N=num_patches, E=embedding_dim
        if self.basename == 'mae':
            assert output.shape[1] == 196, "Num_patches doesn't correspond!"
            assert output.shape[2] == 768, "Embedding dimension doesn't correspond!"
            output = torch.einsum("BHWE->BEHW", output.view(-1, 14, 14, 768)) # 196 = 14 * 14
            self.output_feature['high_level_feature'] = self.high_level_feature(output) # channel dimension for HLF should be 512
            self.output_feature['low_level_feature'] = self.low_level_feature(output) # channel dimension for LLF should be 64
        # END_ADDED

        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']
    

     
#%% Weakly-Supervised localization
class WSL(nn.Module):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=depth, kernel_size=3, padding=1)
        self.cam   = nn.Conv2d(in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn    = nn.BatchNorm2d(depth)
        self.gmp   = nn.AdaptiveMaxPool2d((1,1))
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam     = self.cam(feature)
        logits  = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits
 

#%% Unfiltered Bottleneck layer
class Bottleneck(nn.Module):
    def __init__(self, num_class):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, stride=(2,2), kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn1   = nn.BatchNorm2d(256)
        self.bn2   = nn.BatchNorm2d(num_class)
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.elu(feature)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.elu(feature)
        return feature


#%% Class Activation Guided Attention Mechanism
class CAGAM(nn.Module):    
    def __init__(self, num_tool, num_verb, num_target, in_depth=512):
        super(CAGAM, self).__init__()        
        out_depth               = num_tool        
        self.verb_context       = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)        
        self.verb_query         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_key           = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key      = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_cmap          = nn.Conv2d(in_channels=out_depth, out_channels=num_verb, kernel_size=1)       
        self.target_context     = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)     
        self.target_query       = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_query  = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_key         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_key    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_cmap        = nn.Conv2d(in_channels=out_depth, out_channels=num_target, kernel_size=1)        
        self.gmp       = nn.AdaptiveMaxPool2d((1,1))
        self.elu       = nn.ELU()    
        self.soft      = nn.Softmax(dim=-2)    
        self.flat      = nn.Flatten(2,3)
        self.bn1       = nn.BatchNorm2d(out_depth)
        self.bn2       = nn.BatchNorm2d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm2d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm2d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm2d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth)
        self.bn10      = nn.BatchNorm2d(out_depth) 
        self.bn11      = nn.BatchNorm2d(out_depth) 
        self.bn12      = nn.BatchNorm2d(out_depth)        
        self.encoder_cagam_verb_beta   = torch.nn.Parameter(torch.randn(1))
        self.encoder_cagam_target_beta = torch.nn.Parameter(torch.randn(1))          
                        
    def get_verb(self, raw, cam):
        x  = self.elu(self.bn1(self.verb_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn2(self.verb_query(x)))
        k1 = self.elu(self.bn3(self.verb_key(x)))
        w1 = self.flat(k1).matmul(self.flat(q1).transpose(-1,-2))        
        q2 = self.elu(self.bn4(self.verb_tool_query(cam)))
        k2 = self.elu(self.bn5(self.verb_tool_key(cam)))
        w2 = self.flat(k2).matmul(self.flat(q2).transpose(-1,-2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)
        attention = torch.transpose(attention, -1, -2)
        v = self.flat(z)
        e = (attention.matmul(v) * self.encoder_cagam_verb_beta).reshape(sh)
        e = self.bn6(e + z)
        cmap = self.verb_cmap(e)
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y  
    
    def get_target(self, raw, cam):
        x  = self.elu(self.bn7(self.target_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn8(self.target_query(x)))
        k1 = self.elu(self.bn9(self.target_key(x)))
        w1 = self.flat(k1).transpose(-1,-2).matmul(self.flat(q1))        
        q2 = self.elu(self.bn10(self.target_tool_query(cam)))
        k2 = self.elu(self.bn11(self.target_tool_key(cam)))
        w2 = self.flat(k2).transpose(-1,-2).matmul(self.flat(q2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         
        v = self.flat(z)
        e = (v.matmul(attention) * self.encoder_cagam_target_beta).reshape(sh)
        e = self.bn12(e + z)
        cmap = self.target_cmap(e)
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y
            
    def forward(self, x, cam):
        cam_v, logit_v = self.get_verb(x, cam)
        cam_t, logit_t = self.get_target(x, cam)
        return (cam_v, logit_v), (cam_t, logit_t)

 
#%% Projection function
class Projection(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128):
        super(Projection, self).__init__()
        self.ivt_value = nn.Conv2d(in_channels=num_triplet, out_channels=out_depth, kernel_size=1)
        self.i_value   = nn.Conv2d(in_channels=num_tool, out_channels=out_depth, kernel_size=1)        
        self.v_value   = nn.Conv2d(in_channels=num_verb, out_channels=out_depth, kernel_size=1)
        self.t_value   = nn.Conv2d(in_channels=num_target, out_channels=out_depth, kernel_size=1)       
        self.ivt_query = nn.Linear(in_features=num_triplet, out_features=out_depth)
        self.dropout   = nn.Dropout(p=0.3)
        self.ivt_key   = nn.Linear(in_features=num_triplet, out_features=out_depth)
        self.i_key     = nn.Linear(in_features=num_tool, out_features=out_depth)
        self.v_key     = nn.Linear(in_features=num_verb, out_features=out_depth)
        self.t_key     = nn.Linear(in_features=num_target, out_features=out_depth)
        self.gap       = nn.AdaptiveAvgPool2d((1,1))
        self.elu       = nn.ELU()          
        self.bn1       = nn.BatchNorm1d(out_depth)
        self.bn2       = nn.BatchNorm1d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm1d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm1d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm1d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth) 
        
    def forward(self, cam_i, cam_v, cam_t, X):  
        q = self.elu(self.bn1(self.ivt_query(self.dropout(self.gap(X).squeeze(-1).squeeze(-1)))))  
        k = self.elu(self.bn2(self.ivt_key(self.gap(X).squeeze(-1).squeeze(-1))) )
        v = self.bn3(self.ivt_value(X)) 
        k1 = self.elu(self.bn4(self.i_key(self.gap(cam_i).squeeze(-1).squeeze(-1))) )
        v1 = self.elu(self.bn5(self.i_value(cam_i)) )
        k2 = self.elu(self.bn6(self.v_key(self.gap(cam_v).squeeze(-1).squeeze(-1))))
        v2 = self.elu(self.bn7(self.v_value(cam_v)) )
        k3 = self.elu(self.bn8(self.t_key(self.gap(cam_t).squeeze(-1).squeeze(-1))))
        v3 = self.elu(self.bn9(self.t_value(cam_t)))
        sh = list(v1.shape)
        v  = self.elu(F.interpolate(v, (sh[2],sh[3])))
        X  = self.elu(F.interpolate(X, (sh[2],sh[3])))
        return (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v))


#%% Multi-head of self and cross attention
class MHMA(nn.Module):
    def __init__(self, depth, num_class=100, num_heads=4, use_ln=False):
        super(MHMA, self).__init__()        
        self.concat = nn.Conv2d(in_channels=depth*num_heads, out_channels=num_class, kernel_size=3, padding=1)
        self.bn     = nn.BatchNorm2d(num_class)
        self.ln     = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        self.elu    = nn.ELU()    
        self.soft   = nn.Softmax(dim=-1) 
        self.heads  = num_heads
        
    def scale_dot_product(self, key, value, query):
        dk        = torch.sqrt(torch.tensor(list(key.shape)[-2], dtype=torch.float32))
        affinity  = query.matmul(key.transpose(-1,-2))                        
        attn_w    = affinity / dk        
        attn_w    = self.soft(attn_w)
        attention = attn_w.matmul(value) 
        return attention
    
    def forward(self, inputs):
        (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v)) = inputs     
        query = torch.stack([q]*self.heads, dim=1) # [B,Head,D]
        query = query.unsqueeze(dim=-1) # [B,Head,D,1]        
        key   = torch.stack([k,k1,k2,k3], dim=1) # [B,Head,D]
        key   = key.unsqueeze(dim=-1) # [B,Head,D,1]        
        value = torch.stack([v,v1,v2,v3], dim=1) # [B,Head,D,H,W]
        dims  = list(value.shape) # [B,Head,D,H,W]
        value = value.reshape([-1,dims[1],dims[2],dims[3]*dims[4]])# [B,Head,D,HW]          
        attn  = self.scale_dot_product(key, value, query)  # [B,Head,D,HW]
        attn  = attn.reshape([-1,dims[1]*dims[2],dims[3],dims[4]]) # [B,DHead,H,W]
        mha   = self.elu(self.bn(self.concat(attn)))
        mha   = self.ln(mha + X.clone())  
        return mha

 
#%% Feed-forward layer
class FFN(nn.Module):
    def __init__(self, k, num_class=100, use_ln=False):
        super(FFN, self).__init__()
        def Ignore(x): return x
        self.conv1 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=1) 
        self.elu1  = nn.ELU() 
        self.elu2  = nn.ELU() if k>0 else Ignore     
        self.bn1   = nn.BatchNorm2d(num_class)    
        self.bn2   = nn.BatchNorm2d(num_class)
        self.ln    = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        
    def forward(self, inputs,):
        x  = self.elu1(self.bn1(self.conv1(inputs)))
        x  = self.elu2(self.bn2(self.conv2(x)))
        x  = self.ln(x + inputs.clone())
        return x

 
#%% Classification layer
class Classifier(nn.Module):
    def __init__(self, layer_size, num_class=100):
        super(Classifier, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1,1)) 
        self.mlp = nn.Linear(in_features=num_class, out_features=num_class)     
        
    def forward(self, inputs):
        x = self.gmp(inputs).squeeze(-1).squeeze(-1)
        y = self.mlp(x)
        return y
