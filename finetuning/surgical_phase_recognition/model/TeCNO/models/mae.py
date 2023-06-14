import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime

import sys, os
from pathlib import Path
sys.path.append("./pretraining/mae")
import models_vit
from prepare_mae_model import prepare_mae_model
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

### TWO HEAD MODELS ###


class TwoHeadMAEModel(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadMAEModel, self).__init__()

        if (not hparams.return_mae_optimizer_groups):
            self.model = prepare_mae_model(
                    hparams.mae_model, 
                    hparams.nb_classes, 
                    hparams.drop_path, 
                    'global_pool', 
                    hparams.mae_ckpt, 
                    hparams.freeze_weights,
                    verbose=False,
                    debug=False
                    )
            self.param_groups = None
        else:
            model_dict = prepare_mae_model(
                    hparams.mae_model, 
                    hparams.nb_classes, 
                    hparams.drop_path, 
                    'global_pool', 
                    hparams.mae_ckpt, 
                    hparams.freeze_weights,
                    reinit_n_layers=hparams.mae_reinit_n_layers,
                    return_optimizer_groups=True,
                    weight_decay=hparams.mae_weight_decay,
                    layer_decay=hparams.mae_layer_decay,
                    verbose=False,
                    debug=False
                    )
            
            self.model = model_dict["model"]
            self.param_groups = model_dict["param_groups"]
        
        print('MAE model loaded.')

        # replace final layer with number of labels
        #self.model.head = Identity()
        self.fc_phase = nn.Linear(self.model.num_classes, hparams.out_features)
        self.fc_tool = nn.Linear(self.model.num_classes, hparams.out_features)

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        tool = self.fc_tool(out_stem)
        return out_stem, phase, tool

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        MAEmodel_specific_args = parser.add_argument_group(
            title='MAEmodel specific args options')

        # Model parameters
        MAEmodel_specific_args.add_argument("--mae_model",
                                            default="vit_base_patch16",
                                            choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch16"],
                                            help="MAE architecture to use")

        MAEmodel_specific_args.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')

        # Finetuning params
        MAEmodel_specific_args.add_argument('--mae_ckpt', default='',
                            help='Start training from an MAE checkpoint')

        MAEmodel_specific_args.add_argument('--freeze_weights', type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4, 5], 
            help=
            """
            Whether or not to partially freeze the weights. 
            Setting freeze weights to: 
            -1 --- does not freeze any weights, 
             0 --- freezes everything except last 2 linear layers, 
             i = 1,2,3...n --- leaves the last linear layers and last i attention blocks of the encoder unfrozen.

             Default -1 --- don't freeze anything.
            """)

        MAEmodel_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=128)

        # Last layer parameters
        MAEmodel_specific_args.add_argument('--nb_classes', default=2048, type=int,
                            help='number of the features in the output of the last layer (head)') # set it to 0 to remove the last layer
        
        # LLRD
        MAEmodel_specific_args.add_argument('--mae_layer_decay', type=float, default=1., # setting this to 1 disables it
                            help='layer-wise lr decay from ELECTRA/BEiT')

        # MAE weight_decay
        MAEmodel_specific_args.add_argument('--mae_weight_decay', type=float, default=0.,
                            help="MAE model's weight decay (default: 0.)")

        # re-init MAE weights
        MAEmodel_specific_args.add_argument('--mae_reinit_n_layers', type=int, default=-1,
                            help='Re-initialize last n layers of the encoder.')

        # return MAE optimizer parameter groups
        MAEmodel_specific_args.add_argument('--return_mae_optimizer_groups', action='store_true',
                            help="""
                                    Whether or not to prepare a list of MAE model parameter groups which you can then pass to the optimizer
                                    of your training procedure. The output will be a list of dictionaries with three keys:
                                    1) "params": a list of model parameters belonging to this group
                                    2) "weight_decay": weight decay for the group
                                    3) "lr_scale": scaling to apply to the LR of your training procedure

                                    This automatically sets weight decay for all the parameters (except for linear layer biases and layer 
                                    norm layers). It also applies Layer Wise Learning Rate Decay (LLRD), by calculating a LR scaling which
                                    should be applied to the learning rate before training.
                                    
                                    NOTE: In order to train your model properly, before passing the parameter groups to the optimizer, you 
                                    should add a "lr" key to each group and set it's value to:

                                    for g in optimizer_groups:
                                        g["lr"] = your_initial_LR * g["lr_scale"]
                            """)
        MAEmodel_specific_args.set_defaults(return_mae_optimizer_groups=False)

        return parser

#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
