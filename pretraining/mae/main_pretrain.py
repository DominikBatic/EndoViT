# -----------------------------------------------------------------------------
# We build upon the code of Masked Autoencoders (MAE) which can be found at:
# MAE: https://github.com/facebookresearch/mae
# -----------------------------------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import configargparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_and_validate_one_epoch, evaluate

# check which gpu you are on
import subprocess

# stochastic weight averaging
from torch.optim.swa_utils import AveragedModel, SWALR
# LLRD
import util.lr_decay as lrd

# rand_augmentation
from util.datasets import build_dataset

# wandb logger
import wandb

# math ceil for swalr anneling epochs calculation
import math

# Allows us to create subsets of ImageFolder datasets
from torch.utils.data import Subset

# Subdataset selection
def trim_dataset(dataset, args, is_train=True):
    possible_subfolders = dataset.class_to_idx.keys()

    valid_subfolders = get_valid_folders(args, possible_subfolders, is_train)

    train_val_text = "Train" if is_train else "Validation"

    # select the indices of valid folders
    if (valid_subfolders != possible_subfolders):
        indices = []
        for folder in valid_subfolders:
            idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[folder]]
            indices.extend(idx)

        indices.sort()
        print(f"{train_val_text} subdatasets {valid_subfolders} will be used.")
        return Subset(dataset, indices)
    
    else:
        print(f"All {train_val_text} subdatasets will be used.")
        return dataset


def get_valid_folders(args, possible_folders, is_train):
    valid_folders = []
    invalid_folders = []

    train_val_text = "Train" if is_train else "Validation"

    if (is_train):
        if (not args.train_datasets_to_take): # if args.train_datasets_to_take is an empty list, then take all possible folders
            return possible_folders
    else:
        if (not args.val_datasets_to_take): # if args.val_datasets_to_take is an empty list, then take all possible folders
            return possible_folders

    for folder in (args.train_datasets_to_take if is_train else args.val_datasets_to_take):
        if (folder in possible_folders):
            valid_folders.append(folder)
        else:
            invalid_folders.append(folder)

    if (invalid_folders):
        print(f"WARNING: Following {train_val_text} folders were not found: {invalid_folders}")

    return valid_folders



def get_args_parser():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # all other arguments can be written in this config file
    parser.add_argument('--config', is_config_file=True, help='config file path')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Layer-wise Learning Rate Decay (LLRD)
    # -> set it to 1. to disable it
    parser.add_argument('--layer_decay', type=float, default=1., # setting this to 1 disables it
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Mixup params	
    parser.add_argument('--mixup', type=float, default=0,	
                        help='mixup alpha, mixup enabled if > 0.')	
    parser.add_argument('--cutmix', type=float, default=0,	
                        help='cutmix alpha, cutmix enabled if > 0.')	
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,	
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')	
    parser.add_argument('--mixup_prob', type=float, default=1.0,	
                        help='Probability of performing mixup or cutmix when either/both is enabled')	
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,	
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')	
    parser.add_argument('--mixup_mode', type=str, default='batch',	
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--nb_classes', default=11, type=int,
                        help='Number of classes for mixup. Set it to the number of datasets you have. Note: this argument is just to satisfy the form of mixup_fn, but otherwise is not used.')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_best_model_at', default='./output_dir/best.pth',
                        help="""
                        Path where to save the best checkpoint. Use this variable if you wish to save
                        the best model at a specific location.

                        Note: this should be a full path including the output file name (best.pth by default),
                              therefore you can rename the output file as well
                        """)
    parser.add_argument('--log_dir', default='./output_dir/tensorboard_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # NOTE: we have disabled distributed training from MAE
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--mean', type=str, default='[0.485, 0.456, 0.406]',
                        help='Mean of the pre-training dataset for normalization.')

    parser.add_argument('--std', type=str, default='[0.229, 0.224, 0.225]',
                        help='Standard deviation of the pre-training dataset for normalization.')

    parser.add_argument('--dist_validation', action='store_true', default=False,
                        help='Enabling distributed validation (recommended during training for faster monitor')

    parser.add_argument('--val_data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='validation dataset path')
    
    # Since the datasets are loaded in by torchvision.datasets.ImageFolder there is no option to exclude some subfolders (i.e. in our case datasets).
    # This will allow to manually specify which subfolders/datasets to take, by first loading all the datasets and then calling torch.utils.data.Subset
    # on the full dataset with indicies we wish to take.
    # NOTE: Since we call ImageFolder first, the number of classes loaded in will still be equal to how many datasets you have at --data_path.
    #       Calling .Subset will not remove how many classes were loaded in, just remove all examples of "exluded" classes. This is important
    #       if you wish to do a finetuning task afterwards, e.g. classification.
    parser.add_argument('--train_datasets_to_take', nargs='+', type=str, default='',
                        help="Names of the train dataset subfolders at --data_path you wish to include in the training dataset.")
    parser.add_argument('--val_datasets_to_take', nargs='+', type=str, default='',
                        help="Names of the validation dataset subfolders at --val_data_path you wish to include in the validation dataset.")
    
    parser.add_argument('--log_every_n_steps', type=int, default=100,
                        help=
                        """After how many steps to log to tensorboard during training. Note: if accum_iter > 1.
                        Then logging will occur every accum_iter * log_every_n_steps. I.e. accum_iter steps are counted as 1 step.
                        """)

    parser.add_argument('--validate_every_n_steps', type=int, default=100,
                        help=
                        """
                        This flags allows more frequent validation during training. I.e. not only once at the end of each epoch.
                        During training a validation function will be called after every "validate_every_n_steps" steps.
                        Additionally, validation will still be performed once at the end of each epoch. If you wish to disable
                        the more frequent validation set this flag to 0.
                        """)

    # NOTE: MSE was the best in our preliminary tests
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'l1', 'Mix'], 
        help=
        """
        We allow several loss functions:
        1) (default) Mean Squared Error
        2) Mean Average Error or l1 loss
        3) Mixed loss is the combination of l1 loss and Multi Stage Structural Similarity Index Measure (MS_SSIM).
           We combine them according to the following formula:
                alpha * MS_SSIM + (1 - alpha) * l1,
           where alpha=0.84 according to the paper: "https://arxiv.org/pdf/1511.08861.pdf"
        """
        )
    
    parser.add_argument('--high_pass_filter_loss', action='store_true',
                        help=
                        """
                        Use an additional loss function which applies high pass filter to both predictions and targets.
                        Afterwards, an L2/MSE loss is calculated on their difference.
                        The final loss is:
                            alpha * loss_hp + (1 - alpha) * loss,

                        where 
                        1) "loss" is the standard MSE/L1/Mix loss described,
                        2) "loss_hp" is the MSE loss after applying the high pass filter to both predictions and targets,
                        3) "alpha=0.84" according to the paper: "https://arxiv.org/pdf/1511.08861.pdf".
                        """)
    
    # NOTE: this didn't improve the results
    parser.set_defaults(high_pass_filter_loss=False)

    # stochastic weight averaging (SWA)
    parser.add_argument('--swa', action='store_true',
                        help='Use stochastic weight averaging during training')
    parser.set_defaults(swa=False)

    parser.add_argument('--swa_start', type=int, default=10,
                        help="If applying swa, start swa from this epoch.")
    
    parser.add_argument('--swa_average_every_n_steps', type=int, default=100,
                        help=
                        """
                        This flag sets how often do we average the models while doing swa. E.g. if swa_average_every_n_steps = 100,
                        every 100 steps we will update the swa average model with the new weights. During our tests, we update the
                        average model before every validation. I.e. swa_average_every_n_steps = validate_every_n_steps.
                        """)
    
    parser.add_argument('--swa_lr', type=float, default=5.5e-5,
                        help="If applying swa, what learning rate to use. We will anneal the last LR to swa_lr during one epoch.")

    parser.add_argument('--rand_aug', action='store_true',
                        help='Apply random augmentation from timm library.')
    parser.set_defaults(rand_aug=False)

    # NOTE: this didn't improve the results
    parser.add_argument('--reinit_n_layers', type=int, default=-1,
                        help='Re-initialized last n layers of the encoder and first n-1 layers of the decoder, as well as some inbetween layers (like normalization layers and so on).')

    # wandb loggin info
    parser.add_argument('--use_wandb', action='store_true',
                        help=
                        """
                        Whether or not to use wandb logger in addition to Tensorboard.

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

    # wandb run name
    parser.add_argument('--wandb_run_name', type=str, default='',
                        help=
                        """
                        Run name which will be displayed in the wandb project repo.
                        Useful for better organization.
                        """)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # mean and std from console input
    mean = args.mean.split(', ')
    mean[0] = mean[0][1:]
    mean[2] = mean[2][:-1]

    for i, value in enumerate(mean):
      mean[i] = float(mean[i])

    args.mean = torch.FloatTensor(mean)

    std = args.std.split(', ')
    std[0] = std[0][1:]
    std[2] = std[2][:-1]

    for i, value in enumerate(std):
      std[i] = float(std[i])

    args.std = torch.FloatTensor(std)

    if (args.rand_aug):
            dataset_train = build_dataset(is_train=True, args=args)
            dataset_val = build_dataset(is_train=False, args=args)
    else:
        # simple augmentation
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)
                ])

        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)

        print(dataset_train)

        # validation dataset
        transform_val = transforms.Compose([
            transforms.Resize([args.input_size, args.input_size], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])

        dataset_val = datasets.ImageFolder(args.val_data_path, transform=transform_val)
        print(dataset_val)

    # You can now select datasets to include by setting up --train_datasets_to_take and
    # --val_datasets_to_take.

    dataset_train = trim_dataset(dataset_train, args, is_train=True)
    dataset_val = trim_dataset(dataset_val, args, is_train=False)

    print("-" * 50)
    print(f"Total train #images: {len(dataset_train)}")
    print(f"Total val #images: {len(dataset_val)}")
    print("-" * 50)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        if args.dist_validation:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # ADDED 26.01.2023. swa data sampler
        sampler_swa = torch.utils.data.SequentialSampler(dataset_swa)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # wandb logging
    if (not args.use_wandb):
        os.environ["WANDB_MODE"]="offline"
    
    # log in
    wandb.login()

    date_str = datetime.datetime.now().strftime("%H:%M-%d.%m.%y_")

    # initialize the project
    wandb.init(
        project="EndoViT_Pretraining",
        name="run_{}{}".format(date_str, "__" + args.wandb_run_name),
        tags=["pretraining"]
    )

    # add hyperparameters to the config
    wandb.config.update(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # adding mixup and cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        #print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, loss_type=args.loss, hpf_loss=args.high_pass_filter_loss)

    # weight re-init
    if (model.reinit_possible(args.reinit_n_layers)):
        model.reinit_weights(args.reinit_n_layers)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # taken from finetune script
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

    else:
        print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print(f"Distributed mode: {args.distributed}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    if (args.layer_decay == 1.):
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list={'pos_embed', 'cls_token'} | {"decoder_pos_embed", "mask_token"},
                                            layer_decay=args.layer_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # stochastic weight averaging (SWA)
    swa_model = None
    swa_scheduler = None
    swa_start = -1

    if (args.swa):
        # It turns out swa averages the initial weights of the model created. We will create the model at the start of epoch "swa_start".
        swa_start = args.swa_start

    # track the best val loss and save only the best model
    best_loss = float("inf")
    best_epoch = -1

    best_result = {
        "loss" : best_loss,
        "epoch" : best_epoch,
    }
    
    # swa model will be evaluated separtely
    best_result_swa = {
        "loss" : best_loss,
        "epoch" : best_epoch,
    }

    # print everything at the same place
    print("".join(["*"] * 100))
    print("Enabled settings:")

    if (args.rand_aug):
        print("\t -> Random augmentation enabled.")

    if (model_without_ddp.reinit_possible(args.reinit_n_layers)):
        print(f"\t -> Re-init enabled on {args.reinit_n_layers} layers.")

    if mixup_active:
        print("\t -> Mixup is activated!")

    if (args.layer_decay != 1.):
        print(f"\t -> LLRD: Layer decay enabled with value of {args.layer_decay:.2f}.")

    if (args.swa):
        print(f"\t -> SWA enabled starting from epoch {swa_start + 1}.")

    print("".join(["*"] * 100))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # do swa for current epoch
        if (args.swa):
            args.do_swa = swa_start <= epoch
        else:
            args.do_swa = False

        if (args.do_swa and swa_model is None):
            # creating swa_model and scheduler at the beginning of epoch "swa_start" because it seems initial weights also get averaged.
            print("-" * 50)
            print("-" * 50)
            print("Creating SWA model!")
            print("-" * 50)
            print("-" * 50)
            swa_model = AveragedModel(model).to(device)
            swa_scheduler = SWALR(
                optimizer, 
                anneal_strategy="cos", 
                anneal_epochs=math.ceil(len(data_loader_train) / args.swa_average_every_n_steps), # annealing over 1 epoch
                swa_lr=args.swa_lr
                )

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, val_stats = train_and_validate_one_epoch(
            model, model_without_ddp, data_loader_train, data_loader_val,
            optimizer, device, epoch, loss_scaler,
            args,
            swa_model, swa_scheduler, swa_start,
            log_writer=log_writer,
            best_result_dict=best_result,
            best_result_dict_swa=best_result_swa,
            mixup_fn=mixup_fn
        )

        # Moved this from before the train_and_validate_one_epoch to after. Now epoch % 5 == 0 instead of previously epoch % 5 == 1.
        # By resetting the loss every 5 epochs the best model will be saved in each of the intervals [epoch 0, epoch 1>, [epoch 1, epoch 5], [epoch 6, epoch 10] and so on
        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            src = args.output_dir + '/' + 'best.pth'
            out_name = f'best_ckpt_{best_result["epoch"]}_loss_{best_result["loss"]:.4f}.pth'
            dest = args.output_dir + '/' + out_name

            subprocess.call(['mv', src, dest])

            best_result["loss"] = float("inf")
            best_result["epoch"] = -1

            # save swa model separately
            if (args.do_swa):
                src = args.output_dir + '/' + 'best_swa.pth'
                out_name = f'best_swa_ckpt_{best_result_swa["epoch"]}_loss_{best_result_swa["loss"]:.4f}.pth'
                dest = args.output_dir + '/' + out_name

                subprocess.call(['mv', src, dest])

                best_result_swa["loss"] = float("inf")
                best_result_swa["epoch"] = -1

            # save final model
            if (epoch + 1 == args.epochs):
                if (args.save_best_model_at):
                    # NOTE: If swa is on this will save the swa model regardless if the normal model was better.
                    src = dest
                    dest = args.save_best_model_at

                    # make the output dir if it doesn't exist
                    Path(args.save_best_model_at).parent.mkdir(parents=True, exist_ok=True)
                    subprocess.call(['cp', src, dest])


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'val_{k}': v for k, v in val_stats.items()}, # CHANGED 16.02.2023. f'test_{k}' -> f'val_{k}'
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("{")
                for i, (k, v) in enumerate(log_stats.items()):
                    if ("patches_below" not in k):
                        if ("_lr" in k):
                            f.write(f"\"{k}\": {v:.2e}")
                        elif ("_loss" in k):
                            f.write(f"\"{k}\": {v:.4f}")
                        elif (k == "epoch"):
                            f.write(f"\"{k}\": {v:2d}")
                        elif (k == "n_parameters"):
                            f.write(f"\"{k}\": {v / 1e+6:3.2f} (M)")
                        else:
                            f.write(f"\"{k}\": {v:.4f}")
                        f.write(", " if i+1 != len(log_stats.items()) else "}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.call(['nvidia-smi', '-L'])
    main(args)
