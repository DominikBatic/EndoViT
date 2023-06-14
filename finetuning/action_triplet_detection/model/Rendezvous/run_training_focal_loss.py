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


#%% import libraries
import os
import sys
import time
import torch
import random
#import network
import argparse
import platform
import ivtmetrics # You must "pip install ivtmetrics" to use
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

# for reproducibility
import torch.backends.cudnn as cudnn

import prettytable

# Redirect stdout and stderr to out.txt and err.txt
from contextlib import redirect_stdout, redirect_stderr

# get an MAE model
sys.path.append("./pretraining/mae")
from prepare_mae_model import prepare_mae_model

# get a ResNet model
from simple_models import ResNet18_with_classifier_head, ResNet50_with_classifier_head

# add wandb logger
import wandb

# datetime for version control
from datetime import datetime

# for ceil function
import math

# accuracy calculation
from metric_tracker import AccuracyTracker

# checking scikit-learn and ivtmetrics version:
import sklearn

# printing learning rates for multiple layers
from utils.util import collect_lr_by_layers

# matplotlib plotting of learning rates for different layers
import matplotlib.pyplot as plt

# get mean and std from console input
def mean_and_std_from_args(mean_str, std_str):
    mean = mean_str.split(', ')
    mean[0] = mean[0][1:]
    mean[2] = mean[2][:-1]

    for i, value in enumerate(mean):
      mean[i] = float(mean[i])

    std = std_str.split(', ')
    std[0] = std[0][1:]
    std[2] = std[2][:-1]

    for i, value in enumerate(std):
      std[i] = float(std[i])

    #print('Mean: {} \n Std: {}'.format(mean, std))
    return (mean, std)

# get a green image of shape (h, w, 3)
def green_img(h, w):
    return torch.concat(
        [
        torch.full((h, w, 1), 50.), 
        torch.full((h, w, 1), 205.), 
        torch.full((h, w, 1), 50.)
        ], dim=-1
    ).cpu().numpy()

# get a red image of shape (h, w, 3)
def red_img(h, w):
    return torch.concat(
        [
        torch.full((h, w, 1), 255.), 
        torch.full((h, w, 1), 0.), 
        torch.full((h, w, 1), 0.)
        ], dim=-1
    ).cpu().numpy()


#%% @args parsing
parser = argparse.ArgumentParser()
# model
parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'], help='Model name?')
parser.add_argument('--version', type=str, default="",  help='Model version control (for keeping several versions)')
# In this code not used:
parser.add_argument('--hr_output', action='store_true', help='Whether to use higher resolution output (32x56) or not (8x14). Default: False')
# In this code not used:
parser.add_argument('--use_ln', action='store_true', help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
# In this code not used:
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ')
parser.add_argument('--backbone', type=str, default='mae', choices=['resnet18', 'resnet50', 'mae'], help='Which network to use for feature extraction.')
# job
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test',  action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,  help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/path/to/dataset', help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval', choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval', 'cholect45-crossval'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,  choices=[1,2,3,4,5,], help='The test split in k-fold cross-validation')
# NOT USED in original repo, now it does something:
parser.add_argument('--image_width', type=int, default=448, help='Image width ')  
# NOT USED in original repo, now it does something:
parser.add_argument('--image_height', type=int, default=256, help='Image height ')
# NOT USED:
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')  
# NOT USED:
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
# NOT USED:
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
# NOT USED:
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
# NOT USED:
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*', default=['original', 'vflip', 'hflip', 'contrast', 'rot90'], help='List augumentation styles (see dataloader.py for list of supported styles).')
# hp
parser.add_argument('-b', '--batch', type=int, default=32,  help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100,  help='How many training epochs?')
# NOTE: we keep the next two parameters as nargs for compability resons with some of the original code, this code only uses one LR and warmup
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[5], help='Number of warmup epochs.')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.001], help='Learning rate for the model')
# CHANGED weight_decay from 5e-2 to 1e-5
parser.add_argument('--weight_decay', type=float, default=1e-5,  help='L2 regularization weight decay constant for the model')
# NOT USED:
parser.add_argument('--decay_steps', type=int, default=10,  help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99,  help='Learning rates weight decay rate')
# NOT USED:
parser.add_argument('--momentum', type=float, default=0.95,  help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1,  help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
# device
parser.add_argument('--gpu', type=str, default="0",  help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')

# set seed for reproducibility
parser.add_argument('--seed', default=-1, type=int, help="If seed!=-1, fix the seed to ensure reproducibility.")

# number of workers for dataloaders
parser.add_argument('--num_workers', default=12, type=int, help="Number of workers for dataloaders.")

# output_dir
parser.add_argument('--output_dir', type=str, default="./__checkpoint__",  help='Path to output directory.')

# MAE model hyperparameters
parser.add_argument("--mae_model",
                    default="vit_base_patch16",
                    choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"],
                    help="MAE architecture to use")

parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--mae_ckpt', default='',
                    help=
                    """
                    Start training from an MAE checkpoint.
                    NOTE: if you are already loading from pre-trained Rendezvous checkpoint, leave this empty.
                    """)

parser.add_argument('--freeze_weights', type=int, default=3, choices=[-1, 0, 1, 2, 3, 4, 5], 
                    help=
                    """
                    Whether or not to partially freeze MAE model's weights. 
                    Setting freeze weights to: 
                    -1 --- does not freeze any weights, 
                    0 --- freezes everything except the last normalization and linear layer, 
                    i = 1,2,3...n --- leaves the last normalization/linear layer and last i attention blocks of the encoder unfrozen.

                    Default 3 --- Leave last 3 blocks unfrozen.
                    """)

# Last MAE layer parameters
parser.add_argument('--nb_classes', default=100, type=int,
                    help='number of the features in the output of the last layer (head)') # set it to 0 to remove the last layer

# MAE Optimizer parameters

# LLRD from fine-tuning script
parser.add_argument('--mae_layer_decay', type=float, default=1., # setting this to 1 disables it
                    help='layer-wise lr decay from ELECTRA/BEiT')

parser.add_argument('--plot_layer_wise_learning_rates', action='store_true',
                    help=
                    """
                    Collect Learning Rates during training for each parameter group. And plot them after the training is over.

                    NOTE: this only works if backbone="mae"
                    """)
parser.set_defaults(plot_layer_wise_learning_rates=False)

# MAE weight_decay (for now use same wd for everything)
#parser.add_argument('--mae_weight_decay', type=float, default=0.05,
#                        help="MAE model's weight decay (default: 0.05)")

# re-init MAE weights
parser.add_argument('--reinit_n_layers', type=int, default=-1,
                        help='Re-initialize last n layers of the encoder.')

# return MAE optimizer parameter groups
parser.add_argument('--return_mae_optimizer_groups', action='store_true',
                        help="""
                        Whether or not to set up weight decay and LLRD separately for MAE. This means the 'prepare_mae_model'
                        will return a dictionary with the model itself and a list of optimizer paramter groups for the MAE model. 
                        This list should be passed to the optimizer later on. The list doesn't set the lr itself, but the lr_scale 
                        only. Because of this, before passing the parameter groups to the optimizer, in each group a "lr" parameter
                        should be set by multiplying the desired learning rate by the lr_scale. Weight decay for MAE is the same 
                        one used for the rest of the network.
                        """)
parser.set_defaults(return_mae_optimizer_groups=False)

# MAE Optimizer parameters

# Overfitting on a small subset of the dataset
parser.add_argument('--overfit', action='store_true',
                    help=
                    """
                    Run an overfitting run, in which we try to overfit the model to a small subset of the train dataset.
                    If true, validation and test dataset will be the same as train dataset.
                    """)

parser.add_argument('--overfit_n_samples', type=int, default=100,
                    help="Number of training samples to overfit to, if --overfit flag is True")

# Arguments for the normalization of the dataset (passed to the dataloader script).
# NOTE: Formatting is important when passing in these arguments. The string should correspond to how a python list
#       gets printed. Extra spaces will cause errors. Example is given by the default values.

# NOTE 2: the default values are ImageNet mean and std (In our tests we use cholec80 mean and std.)
parser.add_argument('--dataset_mean', type=str, default='[0.485, 0.456, 0.406]',
                        help='Mean of the pre-training dataset for normalization.')

parser.add_argument('--dataset_std', type=str, default='[0.229, 0.224, 0.225]',
                        help='Standard deviation of the pre-training dataset for normalization.')

# wandb loggin info
parser.add_argument('--use_wandb', action='store_true',
                    help=
                    """
                    Whether or not to use wandb logger.

                    To use wandb logger follow the next few steps:
                    1) make a free wandb account (https://wandb.ai/site)
                    2) install wandb on your environment: pip install wandb -qq
                    3) Log in to your account.
                       This code will automatically log you in if you set WANDB_API_KEY environment
                       variable to your account's authorization key. 
                       (You can find the key here: https://wandb.ai/authorize)
                       i.e. in your bash environment do the following command: 
                                export WANDB_API_KEY=YOUR_API_KEY

                    4) set the flag --use_wand when running the code
                    """)

# wandb project name and tags
parser.add_argument('--wandb_project_name', type=str, default="RDV_overfitting",  help='Name of the wandb project where to log the run.')
parser.add_argument('--wandb_tags', type=str, nargs='*', default=["overfitting"],  help='A list of wandb tags for the run.')

parser.add_argument('--debug_mode', action='store_true',
                    help=
                    """
                    Turn the debug mode on. This will run a simplified version of the training procedure. 
                    In this mode additional information will be printed to std_out. E.g. parameter groups
                    which get passed to the optimizer, learning rate for specific layers during training
                    and so on.
                    """)
parser.set_defaults(debug_mode=False)

# custom loss weights
# NOTE: "no_weighting" gave us best results
parser.add_argument('--loss_weighting', type=str, default="no_weighting", choices=["no_weighting", "linear", "logarithmic"],
                    help =
                    """
                    Apply pre-determined per-class weights in BCEWithLogitsLoss. For each class we have calculated the number of positive and
                    negative examples and set the weight factor as #Negative/#Positive.

                    There are 3 options:
                        1) no_weighting - doesn't apply any weighting in BCEWithLogitsLoss
                        2) linear - applies the pre-computed weights as described above
                        3) logarithmic - sets the weight factor as log(#Negative/#Positive)
                    """)

# data efficient training and printing
parser.add_argument('--data_efficient_training', type=int, nargs='*', default=[], 
                    help=
                    """
                    To test out how our models perform when less training data is available we allow manually selecting a subset of training videos to take
                    as the training dataset. In our tests we will select either 2, 4 or 8 videos. The videos should be specified as a list of integers. 
                    
                    NOTE 1: If you select a video number that is outside the training dataset an error will be raised. 
                    NOTE 2: Leave this list empty to use the full training dataset.
                    """)

FLAGS, unparsed = parser.parse_known_args()

#%% @params definitions
is_train        = FLAGS.train
is_test         = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir        = FLAGS.data_dir
kfold           = FLAGS.kfold if "crossval" in dataset_variant else 0
version         = FLAGS.version
hr_output       = FLAGS.hr_output
use_ln          = FLAGS.use_ln
batch_size      = FLAGS.batch
pretrain_dir    = FLAGS.pretrain_dir
test_ckpt       = FLAGS.test_ckpt
weight_decay    = FLAGS.weight_decay
learning_rates  = FLAGS.initial_learning_rates
warmups         = FLAGS.warmups
decay_steps     = FLAGS.decay_steps
decay_rate      = FLAGS.decay_rate
power           = FLAGS.power
momentum        = FLAGS.momentum
epochs          = FLAGS.epochs
gpu             = FLAGS.gpu
image_height    = FLAGS.image_height
image_width     = FLAGS.image_width
image_channel   = FLAGS.image_channel
num_triplet     = FLAGS.num_triplet_classes
num_tool        = FLAGS.num_tool_classes
num_verb        = FLAGS.num_verb_classes
num_target      = FLAGS.num_target_classes
val_interval    = FLAGS.epochs-1 if FLAGS.val_interval==-1 else FLAGS.val_interval
set_chlg_eval   = True if "challenge" in dataset_variant else False # To observe challenge evaluation protocol
gpu             = ",".join(str(FLAGS.gpu).split(","))
decodelayer     = FLAGS.decoder_layer
addnorm         = "layer" if use_ln else "batch"
modelsize       = "high" if hr_output else "low"
FLAGS.multigpu  = len(gpu) > 1  # not yet implemented !
mheaders        = ["", "", "k"] #["","l", "", "k"]
margs           = [FLAGS.backbone, dataset_variant, kfold]
modelname       = "_".join(["{}{}".format(x,y) for x,y in zip(mheaders, margs) if len(str(y))])

# data efficient training
data_efficient_training = FLAGS.data_efficient_training

# overfitting
overfit         = FLAGS.overfit
overfit_n_samples = FLAGS.overfit_n_samples
output_dir      = FLAGS.output_dir
# normalization hyperparameters
dataset_mean, dataset_std = mean_and_std_from_args(FLAGS.dataset_mean, FLAGS.dataset_std)

current_run_number = "1"
if (os.path.exists(output_dir)):
    runs = os.listdir(output_dir)
    if (runs):
        current_run_number = str(max(map(lambda dir_name : int(dir_name.split("_")[1]) + 1 if dir_name.split("_")[0] == "run" else 1, runs)))

date_str = datetime.now().strftime("%H:%M-%d.%m.%y_")
model_dir       = output_dir + "/run_{}_{}_{}".format(current_run_number.zfill(4), date_str, version)
if not os.path.exists(model_dir): os.makedirs(model_dir)

resume_ckpt     = None
ckpt_path       = os.path.join(model_dir, '{}.pth'.format(modelname))
ckpt_path_best_acc   = os.path.join(model_dir, '{}_best_acc.pth'.format(modelname))
logfile         = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations      = FLAGS.augmentation_list 
iterable_augmentations  = []

# ADDED debug mode
debug_mode = FLAGS.debug_mode

if (debug_mode):
    overfit=True
    overfit_n_samples=100
    epochs=10

# Redirect stdout and stderr to out.txt and err.txt.
# Had to be done from the code itself because of automatic numbering.
if (is_test and not is_train):
    stdout_file_path = model_dir + '/out_test.txt'
    stderr_file_path = model_dir + '/err_test.txt'
else: 
    stdout_file_path = model_dir + '/out.txt'
    stderr_file_path = model_dir + '/err.txt'

print("Redirected stdout to {}".format(stdout_file_path))
print("Redirected stderr to {}".format(stderr_file_path))

with open(stderr_file_path, 'w') as f_err, redirect_stderr(f_err):    
    with open(stdout_file_path, 'w') as f_out, redirect_stdout(f_out):
        # checking scikit-learn and ivtmetrics version:
        print('###### The scikit-learn version is {}. ######'.format(sklearn.__version__))
        print('###### The ivtmetrics version is {}. ######'.format(ivtmetrics.__version__))

        # number of workers for dataloaders
        num_workers_dl = FLAGS.num_workers

        # set seed for reproducibility
        seed = FLAGS.seed
        if (seed != -1):
            print("Setting fixed seed to ensure reproducibility ...")
            print(f"\tseed: {seed}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            cudnn.benchmark = True

        backbone            = FLAGS.backbone
        use_wandb           = FLAGS.use_wandb
        wandb_project_name  = FLAGS.wandb_project_name
        wandb_tags          = FLAGS.wandb_tags

        # MAE model hyperparameters
        mae_hyperparams = {
                "mae_model"                     : FLAGS.mae_model,
                "mae_ckpt"                      : FLAGS.mae_ckpt,
                "drop_path"                     : FLAGS.drop_path,
                "freeze_weights"                : FLAGS.freeze_weights,
                "nb_classes"                    : FLAGS.nb_classes,
                "return_mae_optimizer_groups"   : FLAGS.return_mae_optimizer_groups,
                "mae_layer_decay"               : FLAGS.mae_layer_decay,
                "reinit_n_layers"               : FLAGS.reinit_n_layers,
                }
        
        # wandb logging
        if (not use_wandb):
            os.environ["WANDB_MODE"]="offline"
        
        # log in
        wandb.login()

        # initialize the project
        wandb.init(
            # wandb project name and tags
            project=wandb_project_name,
            name="run_{}_{}_{}".format(current_run_number.zfill(4), date_str, version),
            tags=[backbone] + wandb_tags
        )

        # add hyperparameters to the config
        wandb.config.update(FLAGS)

        # NoPretraining checkpoint can be set by setting mae_ckpt = "No_ckpt"
        if (backbone == "mae" and mae_hyperparams["mae_ckpt"] == "No_ckpt"):
            mae_hyperparams["mae_ckpt"] = ''

        # If we load from an existing Rendezvous checkpoint, MAE model parameters should be there as well.
        # This then prevents MAE parameters from being loaded twice. Once in "prepare_mae_model" function, and then second time when RDV checkpoint gets loaded.
        if (backbone == "mae" and ((is_test and not is_train) or os.path.exists(ckpt_path) or os.path.exists(pretrain_dir))):
            mae_hyperparams["mae_ckpt"] = ''
            print("WARNING: The network will load parameters from existing RDV checkpoint. In this case --mae_ckpt hyperparameter is unnecessary. It will be set to --mae_ckpt='' automatically.")

        # PrettyTable printout of passed parameters.
        print("")
        print("".join(["*"] * 200))
        print("The following parameters were passed in:")

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter Name", "Parameter Value"]
        table.align["Parameter Name"] = "r"
        table.align["Parameter Value"] = "l"
        table.hrules=prettytable.ALL

        for name, value in vars(FLAGS).items():
            table.add_row([name, value])

        print(table)

        print("")
        print("".join(["*"] * 200))
        print("")

        print("Configuring network ...")

        #%% @functions (helpers)
        def assign_gpu(gpu=None):  
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 
            os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' 
            

        def get_weight_balancing(case='cholect50'):
            # 50:   cholecT50, data splits as used in rendezvous paper
            # 50ch: cholecT50, data splits as used in CholecTriplet challenge
            # 45cv: cholecT45, official data splits (cross-val)
            # 50cv: cholecT50, official data splits (cross-val)
            switcher = {
                'cholect50': {
                    'tool'  :   [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
                    'verb'  :   [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636, 12.13692946, 0.40462028],
                    'target':   [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446, 0.285756500, 12.72368421, 0.6250808,  3.85771277, 6.95683453, 0.84923888, 0.40130032]
                },
                'cholect50-challenge': {
                    'tool':     [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
                    'verb':     [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 1.28428410, 7.35822511, 18.67857143, 0.45704490],
                    'target':   [0.07333818, 0.87139287, 0.42853950, 1.00000000, 17.67281106, 13.94545455, 1.44880997, 6.04889590, 0.326188650, 16.82017544, 0.63577586, 6.79964539, 6.19547658, 0.96284208, 0.51559559]
                },
                'cholect45-crossval': {
                    1: {
                        'tool':     [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                        'verb':     [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889, 17.07784431, 0.45241117],
                        'target':   [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319, 0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
                        },
                    2: {
                        'tool':     [0.0854156, 0.89535362, 0.10995253, 2.74936869, 1.78264429, 1.13234529],
                        'verb':     [0.36346863, 0.06771776, 0.07893261, 0.82842725, 1.33892161, 2.13049748, 1.26120359, 5.72674419, 19.7, 0.43189126],
                        'target':   [0.07530655, 0.97961957, 0.4325135, 0.99393438, 15.5387931, 14.5951417, 1.53862569, 6.01836394, 0.35184462, 15.81140351, 0.709506, 5.79581994, 8.08295964, 1.0, 0.52689272]
                    },
                    3: {
                        "tool" :   [0.0915228, 0.89714969, 0.12057004, 2.72128174, 1.94092281, 1.12948557],
                        "verb" :   [0.43636862, 0.07558554, 0.0891017, 0.81820519, 1.53645582, 2.31924198, 1.28565657, 6.49387755, 18.28735632, 0.48676763],
                        "target" : [0.06841828, 0.90980736, 0.38826607, 1.0, 14.3640553, 12.9875, 1.25939394, 5.38341969, 0.29060227, 13.67105263, 0.59168565, 6.58985201, 5.72977941, 0.86824513, 0.47682423]

                    },
                    4: {
                        'tool':     [0.08222218, 0.85414117, 0.10948695, 2.50868784, 1.63235867, 1.20593318],
                        'verb':     [0.41154261, 0.0692142, 0.08427214, 0.79895288, 1.33625219, 2.2624166, 1.35343681, 7.63, 17.84795322, 0.43970609],
                        'target':   [0.07536126, 0.85398445, 0.4085784, 0.95464422, 15.90497738, 18.5978836, 1.55875831, 5.52672956, 0.33700863, 15.41666667, 0.74755423, 5.4921875, 6.11304348, 1.0, 0.50641118],
                    },
                    5: {
                        'tool':     [0.0804654, 0.92271157, 0.10489631, 2.52302243, 1.60074906, 1.09141982],
                        'verb':     [0.50710436, 0.06590258, 0.07981184, 0.81538866, 1.29267277, 2.20525568, 1.29699248, 7.32311321, 25.45081967, 0.46733895],
                        'target':   [0.07119395, 0.87450495, 0.43043372, 0.86465981, 14.01984127, 23.7114094, 1.47577277, 5.81085526, 0.32129865, 22.79354839, 0.63304067, 6.92745098, 5.88833333, 1.0, 0.53175798]
                    }
                },
                'cholect50-crossval': {
                    1:{
                        'tool':     [0.0828851, 0.8876, 0.10830995, 2.93907285, 1.63884786, 1.14499484],
                        'verb':     [0.29628942, 0.07366916, 0.08267971, 0.83155428, 1.25402434, 2.38358209, 1.34938741, 7.56872038, 12.98373984, 0.41502079],
                        'target':   [0.06551745, 1.0, 0.36345711, 0.82434783, 13.06299213, 8.61818182, 1.4017744, 4.62116992, 0.32822238, 45.45205479, 0.67343211, 4.13200498, 8.23325062, 0.88527215, 0.43113306],

                    },
                    2:{
                        'tool':     [0.08586283, 0.87716737, 0.11068887, 2.84210526, 1.81016949, 1.16283571],
                        'verb':     [0.30072757, 0.07275414, 0.08350168, 0.80694143, 1.39209979, 2.22754491, 1.31448763, 6.38931298, 13.89211618, 0.39397505],
                        'target':   [0.07056703, 1.0, 0.39451115, 0.91977006, 15.86206897, 9.68421053, 1.44483706, 5.44378698, 0.31858714, 16.14035088, 0.7238395, 4.20571429, 7.98264642, 0.91360477, 0.43304307],
                    },
                    3:{
                    'tool':      [0.09225068, 0.87856006, 0.12195811, 2.82669323, 1.97710987, 1.1603972],
                        'verb':     [0.34285159, 0.08049804, 0.0928239, 0.80685714, 1.56125608, 2.23984772, 1.31471136, 7.08835341, 12.17241379, 0.43180428],
                        'target':   [0.06919395, 1.0, 0.37532866, 0.9830703, 15.78801843, 8.99212598, 1.27597765, 5.36990596, 0.29177312, 15.02631579, 0.64935557, 5.08308605, 5.86643836, 0.86580743, 0.41908257], 
                    },
                    4:{
                        'tool':     [0.08247885, 0.83095539, 0.11050268, 2.58193042, 1.64497676, 1.25538881],
                        'verb':     [0.31890981, 0.07380354, 0.08804592, 0.79094077, 1.35928144, 2.17017208, 1.42947103, 8.34558824, 13.19767442, 0.40666428],
                        'target':   [0.07777646, 0.95894072, 0.41993829, 0.95592153, 17.85972851, 12.49050633, 1.65701092, 5.74526929, 0.33763901, 17.31140351, 0.83747083, 3.95490982, 6.57833333, 1.0, 0.47139615],
                    },
                    5:{
                        'tool':     [0.07891691, 0.89878025, 0.10267677, 2.53805556, 1.60636428, 1.12691169],
                        'verb':     [0.36420961, 0.06825313, 0.08060635, 0.80956984, 1.30757221, 2.09375, 1.33625848, 7.9009434, 14.1350211, 0.41429631],
                        'target':   [0.07300329, 0.97128713, 0.42084942, 0.8829883, 15.57142857, 19.42574257, 1.56521739, 5.86547085, 0.32732733, 25.31612903, 0.70171674, 4.55220418, 6.13125, 1.0, 0.48528321],
                    }
                }
            }
            return switcher.get(case)
            

        def train_loop(dataloader, model, activation, loss_fn_ivt, optimizer, scheduler, epoch, n_steps_per_epoch, logfile, per_class_weights, **kwargs):
            start = time.time() 

            # printing learning rates for different layers
            lr_list = []

            for batch, (imgs, (y1, y2, y3, y4)) in enumerate(dataloader):
                imgs, y4 = imgs.cuda(), y4.cuda()        # we only output predictions for triplets
                model.train()        
                triplet = model(imgs)
                logit_ivt       = triplet                
                loss_ivt        = loss_fn_ivt(logit_ivt, y4.float())
                # ADDED: Focal Loss
                counter_weight = torch.where(y4.float() == 1.0, (-1 / (per_class_weights + 1e-8)), -1.)
                loss            = torch.mean(loss_ivt * torch.pow(1 - torch.exp(loss_ivt * counter_weight), 3))
                # Backpropagation # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()

                # log to wandb
                wandb.log(
                {
                "train/loss_ivt" : loss.item(),
                "train/epoch"    : (batch + 1 + n_steps_per_epoch * epoch) / n_steps_per_epoch                       
                }, commit=True if (batch + 1 < n_steps_per_epoch) else False)

                
            # learning rate schedule update
            scheduler.step()
            print(f'completed | Losses => ivt: [{loss.item():.4f}] >> eta: {(time.time() - start):.2f} secs', file=open(logfile, 'a+'))

            # printing learning rates for different layers
            if ("collect_LRs" in kwargs and kwargs["collect_LRs"]):
                lr_list.append(collect_lr_by_layers(optimizer))
            
            return lr_list

        def test_loop(dataloader, model, activation, loss_fn_ivt, mAP_tracker, acc_tracker, per_class_weights, is_test = False, last_video=False, print_qualitative_examples=False, epoch=0):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            mAP_tracker.reset()
            acc_tracker.reset()

            total_loss = 0

            with torch.no_grad():
                for batch, (imgs, (y1, y2, y3, y4)) in enumerate(dataloader):
                    imgs, y4 = imgs.cuda(), y4.cuda()         
                    model.eval()  
                    triplet = model(imgs)
                    logit_ivt       = triplet                
                    loss_ivt        = loss_fn_ivt(logit_ivt, y4.float())  
                    # ADDED: Focal Loss
                    counter_weight = torch.where(y4.float() == 1.0, (-1 / (per_class_weights + 1e-8)), -1.)
                    loss            = torch.mean(loss_ivt * torch.pow(1 - torch.exp(loss_ivt * counter_weight), 3))
                    total_loss += loss.item() * imgs.shape[0]

                    targets = y4.float().detach().cpu()
                    preds = activation(triplet).detach().cpu()

                    mAP_tracker.update(targets, preds) # Log metrics

                    preds = torch.where(preds < 0.5, 0., 1.)
                    acc_tracker.update(preds, targets)

                    if (last_video and print_qualitative_examples):
                        if (batch ==  0):
                            if (not is_test):
                                print(f"Epoch {epoch}:\np: {preds[0]}\nt: {targets[0]}")
                            else:
                                print(f"Testing:\np: {preds[0]}\nt: {targets[0]}")

                                tbl = wandb.Table(columns=["Image", "Prediction correct?"])
                                for img, pred, target in zip(imgs, preds, targets):
                                    tbl.add_data(
                                        wandb.Image(torch.permute(img, (1, 2, 0)).detach().cpu().numpy()), 
                                        wandb.Image(green_img(imgs.shape[-2], imgs.shape[-1])) if torch.equal(pred, target) 
                                        else wandb.Image(red_img(imgs.shape[-2], imgs.shape[-1]))
                                    )
                                wandb.log({"Test Table}" : tbl}, commit=False)

            mAP_tracker.video_end()
            acc_tracker.video_end()

            return (total_loss, size)

        def weight_mgt(model, optimizer, scheduler, benchmark_mAP, score_mAP, benchmark_acc, score_mpwAcc, epoch, ckpt_path, ckpt_path_best_acc, logfile):
            # hyperparameter selection based on validation set
            output_text = ""

            if score_mAP > benchmark_mAP[0]:
                torch.save(
                        {
                            "score_mAP": score_mAP,
                            "score_acc": score_mpwAcc,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch + 1
                        }
                        , ckpt_path)
                benchmark_mAP[0] = score_mAP
                print(f'>>> Saving best mAP checkpoint for epoch {epoch} at {ckpt_path}, time {time.ctime()} ', file=open(logfile, 'a+'))

                output_text += "mAP increased!"

            if score_mpwAcc > benchmark_acc[0]:
                torch.save(
                        {
                            "score_mAP": score_mAP,
                            "score_acc": score_mpwAcc,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch + 1
                        }
                        , ckpt_path_best_acc)
                benchmark_acc[0] = score_mpwAcc
                print(f'>>> Saving best mpwAcc checkpoint for epoch {epoch} at {ckpt_path_best_acc}, time {time.ctime()} ', file=open(logfile, 'a+'))  

                output_text += " mpwAcc increased!" if output_text else "mpwAcc increased!"

            return output_text if output_text else "Both mAP and mpwAcc decreased!"


        #%% assign device and set debugger options
        assign_gpu(gpu=gpu)
        np.seterr(divide='ignore', invalid='ignore')
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        # torch.set_deterministic(True)

        #%% data loading : variant and split selection (Note: original paper used different augumentation per epoch)
        
        dataset = dataloader.CholecT50( 
                    dataset_dir=data_dir, 
                    dataset_variant=dataset_variant,
                    test_fold=kfold,
                    augmentation_list=data_augmentations,
                    img_height = image_height,
                    img_width = image_width,
                    normalization_mean=dataset_mean,
                    normalization_std=dataset_std,
                    overfit=overfit,
                    overfit_n_samples=overfit_n_samples,
                    data_efficient_training=data_efficient_training,
                    )

        # build dataset
        train_dataset, val_dataset, test_dataset = dataset.build()

        print("*" * 50)
        print(f"Train dataset: #examples = {len(train_dataset)}")

        total_len = 0
        for video_dataset in val_dataset:
            total_len += len(video_dataset)

        print(f"  Val dataset: #examples = {total_len}")

        total_len = 0
        for video_dataset in test_dataset:
            total_len += len(video_dataset)

        print(f" Test dataset: #examples = {total_len}")
        print("*" * 50)

        # train data loader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=num_workers_dl, pin_memory=True, persistent_workers=True, drop_last=False)
        
        # val dataset is built per video, so load differently
        val_dataloaders = []
        for video_dataset in val_dataset:
            val_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=num_workers_dl, pin_memory=True, persistent_workers=True, drop_last=False)
            val_dataloaders.append(val_dataloader)

        # get number of training steps per epoch for easier logging
        n_steps_per_epoch = len(train_dataloader) #math.ceil(len(train_dataset) / batch_size)

        # test dataset is built per video, so load differently
        test_dataloaders = []
        for video_dataset in test_dataset:
            test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=3*batch_size, num_workers=num_workers_dl, pin_memory=True, persistent_workers=True, drop_last=False)
            test_dataloaders.append(test_dataloader)
        print("Dataset loaded ...")

        #%% class weight balancing
        class_weights = get_weight_balancing(case=dataset_variant)
        if 'crossval' in dataset_variant:
            tool_weight   = class_weights[kfold]['tool']
            verb_weight   = class_weights[kfold]['verb']
            target_weight = class_weights[kfold]['target']
        else:
            tool_weight   = class_weights['tool']
            verb_weight   = class_weights['verb']
            target_weight = class_weights['target']

        # Or constant weights from average of the random sampling of the dataset: we found this to produce better result.
        tool_weight     = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
        verb_weight     = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448, 2.69230769, 0.60062402]
        target_weight   = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034, 0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]


        #%% model
        mae_param_groups = None

        if (backbone == "mae"):
            if (not mae_hyperparams["return_mae_optimizer_groups"]):
                model = prepare_mae_model(
                            mae_hyperparams["mae_model"], # default: vit_base_patch16
                            mae_hyperparams["nb_classes"], # default: 100
                            mae_hyperparams["drop_path"], # default: 0.1
                            'global_pool', 
                            mae_hyperparams["mae_ckpt"], # default: ''
                            mae_hyperparams["freeze_weights"], # default: 3
                            mae_hyperparams["reinit_n_layers"], # default: -1
                            mae_hyperparams["return_mae_optimizer_groups"], # default: False
                            weight_decay, # default 0.
                            mae_hyperparams["mae_layer_decay"], # default 1.0
                            verbose=False,
                            debug=True if debug_mode else False,
                        ).cuda()
            else:
                mae_dict = prepare_mae_model(
                            mae_hyperparams["mae_model"], # default: vit_base_patch16
                            mae_hyperparams["nb_classes"], # default: 100
                            mae_hyperparams["drop_path"], # default: 0.1
                            'global_pool', 
                            mae_hyperparams["mae_ckpt"], # default: ''
                            mae_hyperparams["freeze_weights"], # default: 3
                            mae_hyperparams["reinit_n_layers"], # default: -1
                            mae_hyperparams["return_mae_optimizer_groups"], # default: False
                            weight_decay, # default 0.
                            mae_hyperparams["mae_layer_decay"], # default 1.0
                            verbose=False,
                            debug=True if debug_mode else False,
                            )
                
                model               = mae_dict["model"].cuda()
                mae_param_groups    = mae_dict["param_groups"]


        elif (backbone == 'resnet18'):
            model = ResNet18_with_classifier_head(pretrained=True).cuda()
        elif (backbone == 'resnet50'):
            model = ResNet50_with_classifier_head(pretrained=True).cuda()
        else:
            raise ValueError(f"Backbone {backbone} is not suported.")

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Head has to be properly initialized if you wish to apply Focal Loss.
        if (backbone == "mae" or backbone == "resnet18"):
            torch.nn.init.normal_(model.head.weight, mean=0., std=0.0001)
            torch.nn.init.constant_(model.head.bias, -2.)
        elif (backbone == "resnet50"):
            torch.nn.init.normal_(model.head[1].weight, mean=0., std=0.0001)
            torch.nn.init.constant_(model.head[1].bias, -2.)

        # model parameter printing
        print("")
        print("".join(["*"] * 200))
        print("Model:")

        print(model)
        print('number of params (M): %.2f' % (pytorch_total_params / 1.e6))
        print('number of trainable params (M): %.2f' % (pytorch_train_params / 1.e6))

        print("")
        print("".join(["*"] * 200))
        print("")


        #%% performance tracker for hp tuning
        benchmark = [0.0]
        benchmark_acc = [0.0]
        print("Model built ...")


        #%% Loss
        activation  = nn.Sigmoid().cuda()

        # pre-computed weights
        per_class_weights = [863.6282, 155.4756, 2174.5161, 886.3816, 184.2775, 7492.4444, 1433.9149, 41.2563, 8429.1250, 1925.8857, 337.8995, 5186.7692, 21.1118, 694.2680, 1.0000, 1643.9024, 61.3876, 1.1054, 116.4930, 6.6360, 16.0650, 288.4464, 229.1741, 2496.8148, 3064.5000, 1203.3036, 1247.9074, 380.0226, 302.7883, 35.2781, 256.4084, 1.0000, 1.0000, 732.0543, 590.5877, 3966.1176, 415.3025, 1.0000, 8429.1250, 708.9053, 1531.7500, 8429.1250, 7492.4444, 2106.5312, 530.0315, 1247.9074, 6743.1000, 6743.1000, 1982.5588, 4495.0667, 1.0000, 428.5605, 525.8828, 962.4429, 1.0000, 1.0000, 3210.4762, 27.5404, 12.3494, 26.1392, 2.3176, 20.5123, 254.4583, 180.7817, 635.2358, 3966.1176, 434.1032, 3210.4762, 146.5733, 129.6996, 4816.2143, 748.3444, 1.0000, 1203.3036, 5619.0833, 1.0000, 724.1720, 1821.7297, 89.0414, 50.1304, 5186.7692, 2809.0417, 30.9777, 1.0000, 1247.9074, 6743.1000, 2324.5517, 673.4100, 242.4693, 11239.1666, 874.8571, 4495.0667, 359.6471, 801.8690, 21.4429, 176.4763, 32.6532, 428.5605, 264.5157, 167.6025]
        per_class_weights = torch.FloatTensor(per_class_weights).cuda()
        per_class_weights.requires_grad = False

        assert per_class_weights.shape == torch.Size([100]), f"Expected custom weights to have shape: torch.Size([100]). Instead got {per_class_weights.shape}."

        print("-" * 50)
        if (FLAGS.loss_weighting == "no_weighting"):
            print("No loss weighting will be used.")
            loss_fn_ivt = nn.BCEWithLogitsLoss(reduction='none').cuda()
            per_class_weights = torch.ones(100, dtype=torch.float32).cuda()
        elif (FLAGS.loss_weighting == "logarithmic"):
            print("Logarithmic loss weighting will be used.")
            loss_fn_ivt = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.log(per_class_weights)).cuda()
        else:
            print("Linear loss weighting will be used.")
            loss_fn_ivt = nn.BCEWithLogitsLoss(reduction='none', pos_weight=per_class_weights).cuda()
        print("-" * 50)

        #%% evaluation metrics
        mAP = ivtmetrics.Recognition(100)
        mAP.reset_global()
        print("Metrics built ...")

        #%% optimizer and lr scheduler
        # We use one learning rate for the whole model. For compability reasons with some original RDV code we keep the learning rate in a list.
        # This code keeps the LR constant during the whole training.

        wp_lr               = [lr for lr in learning_rates]

        opt_parameters      = [] # list of parameters to be passed to the optimizer
                                 # each entry is of the following form: {"params": parameters_from_some_part_of_the_model, "lr": lr, "weight_decay": 0.0}

        # mae optimizer parameter groups
        if (mae_param_groups is not None):
            for group in mae_param_groups:
                group["lr"] = wp_lr[0] * group["lr_scale"]

            opt_parameters.extend(mae_param_groups)
        else:
            module          = model.parameters()

            opt_parameters.extend([
                {"params": module, "lr": wp_lr[0], "weight_decay": weight_decay},
            ])


        optimizer            = torch.optim.AdamW(opt_parameters, lr=wp_lr[0], weight_decay=weight_decay)
        # This code keeps the LR constant. 
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=0)

        #%% checkpoints/weights

        start_epoch = 0

        # CHANGED: load from pretrain_dir first, if it doesn't exist then load from ckpt_path (was opposite before)
        if os.path.exists(pretrain_dir):
            print(f"Loading checkpoint from pretrain_dir= {pretrain_dir}")
            checkpoint = torch.load(pretrain_dir)
            pretrained_dict = checkpoint["model_state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            scheduler_state_dict = checkpoint["scheduler_state_dict"]
            score_mAP = checkpoint["score_mAP"]
            score_acc = checkpoint["score_acc"]
            print(f"##### Loaded checkpoint's mAP score: {score_mAP} #####")
            print(f"##### Loaded checkpoint's acc score: {score_acc} #####")

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model.state_dict().update(pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)
            resume_ckpt = pretrain_dir

            benchmark = [score_mAP]
            benchmark_acc = [score_acc]
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)
            start_epoch = checkpoint["epoch"]

        elif os.path.exists(ckpt_path):
            print(f"Path ckpt_path= {ckpt_path} already exists!")
            print("Training will continue where it left off.")
            checkpoint = torch.load(ckpt_path)
            pretrained_dict = checkpoint["state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            scheduler_state_dict = checkpoint["scheduler_state_dict"]
            score_mAP = checkpoint["score_mAP"]
            score_acc = checkpoint["score_acc"]
            print(f"##### Loaded checkpoint's mAP score: {score_mAP} #####")
            print(f"##### Loaded checkpoint's acc score: {score_acc} #####")

            model.load_state_dict(pretrain_dir)
            resume_ckpt = ckpt_path

            benchmark = [score_mAP]
            benchmark_acc = [score_acc]
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)
            start_epoch = checkpoint["epoch"]

        print("Model's weight loaded ...")


        #%% log config
        header1 = "** Run: {} | Framework: PyTorch | Method: {} | Version: {} | Data: CholecT50 | Batch: {} **".format(os.path.basename(__file__), modelname, version, batch_size)
        header2 = "** Time: {} | Start: {}-epoch  {}-steps | Init CKPT: {} | Save CKPT: {} **".format(time.ctime(), 0, 0, resume_ckpt, ckpt_path)
        header3 = "** LR Config: Init: {} | Peak: {} | Warmup Epoch: {} | Rise: {} | Decay {} | train params {} | all params {} **".format([float(f'{scheduler.get_last_lr()[0]:.6f}')], [float(f'{v:.6f}') for v in wp_lr], warmups, power, decay_rate, pytorch_train_params, pytorch_total_params)
        maxlen  = max(len(header1), len(header2), len(header3))
        header1 = "{}{}{}".format('*'*((maxlen-len(header1))//2+1), header1, '*'*((maxlen-len(header1))//2+1) )
        header2 = "{}{}{}".format('*'*((maxlen-len(header2))//2+1), header2, '*'*((maxlen-len(header2))//2+1) )
        header3 = "{}{}{}".format('*'*((maxlen-len(header3))//2+1), header3, '*'*((maxlen-len(header3))//2+1) )
        maxlen  = max(len(header1), len(header2), len(header3))
        print("\n\n\n{}\n{}\n{}\n{}\n{}".format("*"*maxlen, header1, header2, header3, "*"*maxlen), file=open(logfile, 'a+'))
        print("Experiment started ...\n   logging outputs to: ", logfile)

        lr_list_all_epochs = []
        plot_LRs = backbone == "mae" and FLAGS.plot_layer_wise_learning_rates and mae_hyperparams["mae_layer_decay"] != 1.0

        acc_tracker_val = AccuracyTracker()
        #%% run
        if is_train:
            for epoch in range(start_epoch,epochs):
                try:
                    # Train
                    print("Traning | lr: {} | epoch {}".format([scheduler.get_last_lr()[-1]], epoch), end=" | ", file=open(logfile, 'a+'))
                    # lr tracking for different layers
                    if (plot_LRs):
                        lr_list_current_epoch = train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizer, scheduler, epoch, n_steps_per_epoch, logfile, per_class_weights, collect_LRs=plot_LRs)
                        lr_list_all_epochs.extend(lr_list_current_epoch)
                    else:
                        train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizer, scheduler, epoch, n_steps_per_epoch, logfile, per_class_weights)

                    # val
                    if epoch % val_interval == 0:
                        start = time.time()  
                        mAP.reset_global()
                        acc_tracker_val.global_reset()

                        total_loss = 0
                        total_count = 0

                        print("Evaluating @ epoch: ", epoch, file=open(logfile, 'a+'))
                        for i, val_dataloader in enumerate(val_dataloaders):
                            loss, count = test_loop(val_dataloader, model, activation, loss_fn_ivt, mAP, acc_tracker_val, per_class_weights, is_test=False, last_video=True if i+1 == len(val_dataloaders) else False, print_qualitative_examples=True, epoch=epoch)
                            total_loss += loss
                            total_count += count

                        # ADDED: Logging validation mean accuracies and AP per-video
                        results = acc_tracker_val.get_mean_acc_per_video()

                        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                
                        # log to wandb
                        wandb.log(
                        {
                        "val/loss_ivt"  : total_loss / total_count,
                        "val/mAP_ivt"   : mAP_ivt['mAP'],
                        "val/mean_element_wise_acc" : results["mean_element_wise_acc"],
                        "val/mean_prediction_wise_acc" : results["mean_prediction_wise_acc"]
                        }, commit=True)

                        behaviour = weight_mgt(model, optimizer, scheduler, benchmark, float(mAP.compute_video_AP()['mAP'].item(0)), benchmark_acc, results["mean_prediction_wise_acc"], epoch, ckpt_path, ckpt_path_best_acc, logfile)
                        print("\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] | mpwAcc: [{:.4f}] | mewAcc: [{:.4f}]".format(
                            (time.time() - start),
                            mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP'],
                            results['mean_prediction_wise_acc'],
                            results['mean_element_wise_acc']
                        ), file=open(logfile, 'a+'))
                except KeyboardInterrupt:
                    print(f'>> Process cancelled by user at {time.ctime()}, ...', file=open(logfile, 'a+'))   
                    sys.exit(1)
            test_ckpt = ckpt_path

        # ADDED: plot learning rates for different layers with matplotlib
        if (plot_LRs):
            keys = [f"Layer {i:2d}" for i in range(len(model.blocks) + 2)]
            title = "Per layer Learning Rates"
            xname = "epoch"
        
            fig, ax = plt.subplots(figsize=(10, 10))
            x_axis = [i for i in range(len(lr_list_all_epochs))]
            y_axes = [[step[i] for step in lr_list_all_epochs] for i in range(len(keys))]
            ax.set_xlim(0, epochs)
            ax.set_ylim(0, wp_lr[0] + wp_lr[0] / 10)

            colours = 'bgkyr'
            ax.set_title(title)
            ax.set_xlabel(xname)
            ax.set_ylabel("learning rate")

            for i in range(len(y_axes)):
                colour_index = int(i / 3) % 5
                ax.plot(x_axis, y_axes[i], colours[colour_index], label=f"Layer {i:2d}")

            ax.legend(keys)

            wandb.log({"per_layer_lr" : wandb.Image(fig)}, commit=True)

        #%% eval
        if is_test:
            print("Test weight used: ", test_ckpt)
            if (not os.path.exists(test_ckpt)):
                print("ERROR: test_ckpt doesn't exist. Shutting down ...")
                exit(1)

            checkpoint = torch.load(test_ckpt)
            test_dict = checkpoint["model_state_dict"]

            # ADDED: accuracy tracking
            acc_tracker_test = AccuracyTracker()

            model.load_state_dict(test_dict)
            mAP.reset_global()
            acc_tracker_test.global_reset()

            print(''.join(["-"]*300))

            total_loss = 0
            total_count = 0

            for i, test_dataloader in enumerate(test_dataloaders):
                loss, count = test_loop(test_dataloader, model, activation, loss_fn_ivt, mAP, acc_tracker_test, per_class_weights, is_test=True, last_video=True if i+1 == len(test_dataloaders) else False, print_qualitative_examples=True)

                total_loss += loss
                total_count += count
            #--------
            # get mean accuracy over all videos for logging
            results = acc_tracker_test.get_mean_acc_per_video()

            mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
            mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
            mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval) 

            # ADDED: Wandb log
            wandb.log(
            {
            "test/loss_ivt" : total_loss / total_count,
            "test/mAP_iv"   : mAP_iv["mAP"],
            "test/mAP_it"   : mAP_it["mAP"],
            "test/mAP_ivt"  : mAP_ivt["mAP"],
            "test/mean_element_wise_acc" : results["mean_element_wise_acc"],
            "test/mean_prediction_wise_acc" : results["mean_prediction_wise_acc"]
            }, commit=True)

            print('-'*50, file=open(logfile, 'a+'))
            print('Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
            print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
            print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
            print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
            print('-'*50, file=open(logfile, 'a+'))
            print(f'Mean AP:  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
            print(f':::::: : {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ', file=open(logfile, 'a+'))

            # ADDED: Mean per-video acc
            print('-'*50, file=open(logfile, 'a+'))
            print(f"Testing:\n\tmean_element_wise_acc: {results['mean_element_wise_acc']:.4f}\n\tmean_prediction_wise_acc: {results['mean_prediction_wise_acc']:.4f}", file=open(logfile, 'a+'))
            print('='*50, file=open(logfile, 'a+'))
            print("Test results saved @ ", logfile)

        #%% End
        print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-"*maxlen) , file=open(logfile, 'a+'))

        wandb.finish()