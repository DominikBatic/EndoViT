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
import math
import sys
from typing import Iterable, Optional, Union

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from torch.optim.swa_utils import update_bn
from timm.data.mixup import Mixup
import wandb

import util.misc as misc
import util.lr_sched as lr_sched

print_steps = [50, 100, 150]

# for plotting images to tensorboard
def show_image(args, axis, image, title='', alpha=1):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    axis.set_xticks([])
    axis.set_yticks([])
    axis.imshow(torch.clip((image * args.std + args.mean) * 255, 0, 255).int())
    axis.set_title(title, fontsize=6, alpha=alpha)
    return


def show_image_for_loss(axis, image, title='', alpha=1., norm=mpl.colors.Normalize(), cmap='viridis'):
    # image is [H, W]

    assert image.ndim == 2
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.imshow(image, alpha=alpha, norm=norm, cmap=cmap)
    axis.set_title(title, size=6)

    plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    spacing='proportional')

    return


def plot_results(args, images, loss_per_patch):
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(num=1)
    fig.clf()
    fig, axisdict = plt.subplot_mosaic([['1', '2', '3', '4'],
                                    ['5', '5', '6', '6']],
                                    figsize=(15, 15), layout="tight", num=1)

    # plot result images
    figA_ax1 = axisdict['1']
    show_image(args, figA_ax1, images[0], title="original")

    figA_ax2 = axisdict['2']
    show_image(args, figA_ax2, images[1], title="masked")

    figA_ax3 = axisdict['3']
    show_image(args, figA_ax3, images[2], title="reconstruction, loss: {:.2f}".format(loss_per_patch.mean().item()))

    figA_ax4 = axisdict['4']
    show_image(args, figA_ax4, images[3], title="reconstruction + visible")

    # plot loss images

    # set up cmap and norm
    cmap = mpl.cm.seismic
    bounds = [0.01, 0.05, 0.1, 0.3, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    figB_ax1 = axisdict['5']
    show_image_for_loss(figB_ax1, loss_per_patch, title="LossPerPatch", cmap=cmap, norm=norm)

    figB_ax2 = axisdict['6']
    show_image(args, figB_ax2, images[3], title='')
    show_image_for_loss(figB_ax2, loss_per_patch, title="LossPerPatchOverlayed", cmap=cmap, norm=norm, alpha=0.3)

    return fig


# for plotting loss_per_patch
def unpatchify_for_loss(x, patch_size):
    """
    x: (N, L, patch_size**2)
    imgs: (N, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p))
    x = torch.einsum('nhwpq->nhpwq', x)
    imgs = x.reshape(shape=(x.shape[0], h * p, h * p))
    return imgs

def train_and_validate_one_epoch(
                    model: torch.nn.Module,
                    model_without_ddp : torch.nn.Module,
                    data_loader_train: Iterable,
                    data_loader_val: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    swa_model, swa_scheduler, swa_start,
                    log_writer=None,
                    best_result_dict={"loss": float("inf"), "epoch": -1},
                    best_result_dict_swa={"loss": float("inf"), "epoch": -1},
                    mixup_fn: Optional[Mixup] = None,
                    ):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    # log min and max lr opposed to just max
    metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    train_stats = {}
    val_stats = {}
    val_stats_swa = {}

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
        # validation stats have to be reset at the start of every step
        val_stats = {}
        val_stats_swa = {}

        # use swa scheduler
        if args.do_swa:
            if (data_iter_step + 1) % args.swa_average_every_n_steps == 0 or (data_iter_step + 1) == len(data_loader_train): # ADDED 10.02.2023. update the parameters before evaluation only (instead of at every step)
                swa_model.update_parameters(model)  # To update parameters of the averaged model.
                swa_scheduler.step()                # Switch to SWALR.
                # this makes SWA work with LLRD
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
    
        # we use a per iteration (instead of per epoch) lr scheduler
        elif data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # mixup and cutmix
        if mixup_fn is not None:
            samples, _ = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            losses, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
            loss, loss_per_patch = losses

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # log min and max lr opposed to just max
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(max_lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % (accum_iter * args.log_every_n_steps) == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
            log_writer.add_scalar('train/train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

        # wandb logging
        if (args.use_wandb and (data_iter_step + 1) % (accum_iter * args.log_every_n_steps) == 0):
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)

            wandb.log(
                {
                "train/train_loss": loss_value_reduce,
                "train/epoch_1000x": epoch_1000x,
                "train/max_lr": max_lr,
                "train/min_lr": min_lr,
                }, 
                # Commit if there is nothing more to log at this step, i.e. if validation won't be done.
                # We validate if we are at the end of the epoch or frequent evaluation is on and we are at one of the frequent
                # evaluation steps.
                commit=True if not (
                (data_iter_step + 1) == len(data_loader_train) or
                (args.validate_every_n_steps > 0 and (data_iter_step + 1) % args.validate_every_n_steps == 0 and (data_iter_step + 1) != len(data_loader_train)) 
                ) else False
            )

        ###################################################### VALIDATION ######################################################

        # We always validate once per epoch, additionally it is possible to validate more times per epoch by making args.validate_every_n_steps > 0 and < len(data_loader_train).
        # By setting args.validate_every_n_steps=-1 you can disable the more frequent evaluation.

        # We save the best model overall, and every 2 epochs we save the best model from that epoch.
        # NOTE: If swa is on, then we will save both the normal model as well as the swa version.
        epoch_end = False

        # do validation at the end of the epoch
        if ((data_iter_step + 1) == len(data_loader_train)):
            epoch_end = True
            val_stats = evaluate(data_loader_val, model, device, log_writer, epoch, args, epoch_end=epoch_end, is_swa=False)

            # evalute SWA model separately
            if (args.do_swa):
                    val_stats_swa = evaluate(data_loader_val, swa_model, device, log_writer, epoch, args, epoch_end=epoch_end, is_swa=True)

        # do validation every n steps instead of just on epoch end
        if (args.validate_every_n_steps > 0):
            if ((data_iter_step + 1) % args.validate_every_n_steps == 0 and not epoch_end):
                val_stats = evaluate(
                    data_loader_val,
                    model,
                    device,
                    log_writer,
                    epoch,
                    args,
                    epoch_end=epoch_end,
                    is_swa=False,
                    step=data_iter_step,
                    total_steps=len(data_loader_train))
                
                if (args.do_swa):
                    val_stats_swa = evaluate(
                        data_loader_val,
                        swa_model,
                        device,
                        log_writer,
                        epoch,
                        args,
                        epoch_end=epoch_end,
                        is_swa=True,
                        step=data_iter_step,
                        total_steps=len(data_loader_train))

        # save the model if validation stats improved
        if (val_stats):
            save_model_based_on_validation_results(
                model, 
                optimizer, 
                loss_scaler, 
                val_stats, 
                best_result_dict, 
                args, 
                epoch, 
                epoch_end=epoch_end,
                is_swa=False)
            
            if (args.do_swa and val_stats_swa):
                save_model_based_on_validation_results(
                    swa_model.module, # swa_model has the original model saved as .module
                    optimizer, 
                    loss_scaler, 
                    val_stats_swa, 
                    best_result_dict_swa, 
                    args, 
                    epoch, 
                    epoch_end=epoch_end,
                    is_swa=True)

            # log validation stats
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
            
            log_stats(log_writer, val_stats, epoch_1000x, args, epoch_end=epoch_end, is_swa=False)
            if (args.do_swa and val_stats_swa):
                log_stats(log_writer, val_stats_swa, epoch_1000x, args, epoch_end=epoch_end, is_swa=True)

        ###################################################### VALIDATION ######################################################

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_stats.update(val_stats_swa)
    return (train_stats, val_stats)

# from engine_finetune.py
@torch.no_grad()
def evaluate(data_loader, model, device, log_writer, epoch, args, epoch_end=False, is_swa=False, **kwargs):
    metric_logger = misc.MetricLogger(delimiter="  ")

    swa_text = "SWA_" if is_swa else ""

    if (epoch_end):
        header = f'###### Epoch: [{epoch}] {swa_text}Validation ######'
    else:
        header = f'{swa_text}Validation at epoch [{epoch}] step: [{kwargs["step"]}/{kwargs["total_steps"]}]'

    # switch to evaluation mode
    model.eval()

    if (not epoch_end):
        for i, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            images = images.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                losses, y, mask, latent = model(images, mask_ratio=args.mask_ratio)

            loss, loss_per_patch = losses

            if (not is_swa):
                metric_logger.update(loss=loss.item())
            else:
                metric_logger.update(loss_swa=loss.item())

    # plot embeddings with tensorboard at the end of the epoch
    else:
        embeddings = []
        imgs_to_plot = []

        dataset_mean = args.mean.cpu().view(1, 3, 1, 1)
        dataset_std = args.std.cpu().view(1, 3, 1, 1)

        # get the baseline model/module depending on whether we use args.distributed and/or swa
        if (is_swa):
            base_model = model.module.module if args.distributed else model.module
        else:
            base_model = model.module if args.distributed else model

        for i, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            images = images.to(device, non_blocking=True)

            # save a few outputs to tensorboard
            with torch.cuda.amp.autocast():
                losses, y, mask, latent = model(images, mask_ratio=args.mask_ratio)

            loss, loss_per_patch = losses

            # calculate loss_per_patch statistics
            loss_per_patch = loss_per_patch.detach().cpu()

            patches_below_0_3 = torch.where(loss_per_patch < 0.3, 1, 0).sum() / args.batch_size
            patches_below_0_1 = torch.where(loss_per_patch < 0.1, 1, 0).sum() / args.batch_size
            patches_below_0_0_5 = torch.where(loss_per_patch < 0.05, 1, 0).sum() / args.batch_size
            patches_below_0_0_1 = torch.where(loss_per_patch < 0.01, 1, 0).sum() / args.batch_size

            if (not is_swa):
                metric_logger.update(
                    patches_below_0_3=patches_below_0_3, 
                    patches_below_0_1=patches_below_0_1, 
                    patches_below_0_0_5=patches_below_0_0_5, 
                    patches_below_0_0_1=patches_below_0_0_1
                    )
            else:
                metric_logger.update(
                    patches_below_0_3_swa=patches_below_0_3, 
                    patches_below_0_1_swa=patches_below_0_1, 
                    patches_below_0_0_5_swa=patches_below_0_0_5, 
                    patches_below_0_0_1_swa=patches_below_0_0_1
                )

            # plot embeddings with tensorboard
            if i == 0:
                embeddings.append(torch.mean(latent[:, 1:, :], dim=1))
                imgs_to_plot.append(torch.clone(images).detach().cpu() * dataset_std + dataset_mean)

            if i in print_steps:
                images = images.detach().cpu()

                patch_size = base_model.patch_embed.patch_size[0]
                y = base_model.unpatchify(y)
                y = torch.einsum('nchw->nhwc', y).detach().cpu()

                # prepare loss_per_patch for visualization
                loss_per_patch = loss_per_patch.unsqueeze(-1).repeat(1, 1, patch_size ** 2)  # (N, L, p*p)
                loss_per_patch = unpatchify_for_loss(loss_per_patch, patch_size).detach().cpu() # (N, H, W)

                # prepare mask for visualization
                mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)  # (N, L, p*p*3)
                mask = base_model.unpatchify(mask) # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

                images = torch.einsum('nchw->nhwc', images)

                # mask loss_per_patch -> now not a tensor but a np.masked_array
                loss_per_patch = ma.array(loss_per_patch, mask=1 - mask[:, :, :, 0])  # we only take 1 channel

                # masked image
                im_masked = images * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = images * (1 - mask) + y * mask
                index_to_take = 0

                figure_to_add = plot_results(args, [images[index_to_take], im_masked[index_to_take], y[index_to_take], im_paste[index_to_take]], loss_per_patch[index_to_take])

                log_writer.add_figure(f'val/Example_{str(i)}_{swa_text}Results',
                                    figure_to_add,
                                    global_step=epoch)
                
                # wandb logging
                if (args.use_wandb):
                    wandb.log(
                        {f"val/Example_{str(i)}_{swa_text}Results": wandb.Image(figure_to_add)},
                        commit=False # Never commit here because if we are at the epoch end, we have one more wandb.log call.
                    )

            if (not is_swa):
                metric_logger.update(loss=loss.item())
            else:
                metric_logger.update(loss_swa=loss.item())

        # plot embeddings with tensorboard
        log_writer.add_embedding(torch.cat(embeddings, dim=0),
                                label_img=torch.cat(imgs_to_plot, dim=0),
                                global_step=epoch,
                                tag=f"{swa_text}Embedding_Visualization")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* {swa_text}Validation loss {losses.global_avg:.4f}'.format(losses=metric_logger.loss if not is_swa else metric_logger.loss_swa, swa_text=swa_text))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Pushed model saving into a separate function to allow saving both swa and normal model separately.
def save_model_based_on_validation_results(model, optimizer, loss_scaler, val_stats, best_result_dict, args, epoch, epoch_end=False, is_swa=False):
    if args.output_dir:
        # If we are going to save, get the model which is going to be saved. We save when the loss improves, and at the end of certain epochs.
        if (best_result_dict["loss"] > val_stats['loss' if not is_swa else 'loss_swa'] or (epoch_end and (epoch % 2 == 0 or epoch + 1 == args.epochs))):
            if (args.distributed):
                model_to_save = model
                model_to_save_without_ddp = model.module
            else:
                model_to_save = model_to_save_without_ddp = model

            if (best_result_dict["loss"] > val_stats['loss' if not is_swa else 'loss_swa']):
                model_name = "best" if not is_swa else "best_swa"
                misc.save_model(
                    args=args, model=model_to_save, model_without_ddp=model_to_save_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, output_file_name=model_name)

                best_result_dict["loss"] = val_stats['loss' if not is_swa else 'loss_swa']
                best_result_dict["epoch"] = epoch

                print(f"Best checkpoint saved! Epoch: {epoch} Val Loss: {val_stats['loss' if not is_swa else 'loss_swa']:.4f}" if not is_swa else
                  f"Best SWA checkpoint saved! Epoch: {epoch} Val Loss: {val_stats['loss' if not is_swa else 'loss_swa']:.4f}")
                
            # save a model at the end of epochs 0, 2, 4, 6 ... (to compare results without best loss evalutaion)
            if (epoch_end and (epoch % 2 == 0 or epoch + 1 == args.epochs)):
                model_name = ('checkpoint-%s' if not is_swa else 'checkpoint-%s_swa') % epoch
                misc.save_model(
                    args=args, model=model_to_save, model_without_ddp=model_to_save_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, output_file_name=model_name)


# Pushed logging into a separate function to allow logging both swa and normal stats separately.
def log_stats(log_writer, val_stats, epoch_1000x, args, epoch_end=False, is_swa=False):
    # log the test stats in both cases
    suffix = "" if not is_swa else "_swa"

    if log_writer is not None:
        log_writer.add_scalar(f'val/val_loss{suffix}', val_stats['loss' if not is_swa else 'loss_swa'], epoch_1000x)

        if (epoch_end):
            log_writer.add_scalar(f'val/patches_below_0_3{suffix}', val_stats['patches_below_0_3' if not is_swa else 'patches_below_0_3_swa'], epoch_1000x)
            log_writer.add_scalar(f'val/patches_below_0_1{suffix}', val_stats['patches_below_0_1' if not is_swa else 'patches_below_0_1_swa'], epoch_1000x)
            log_writer.add_scalar(f'val/patches_below_0_0_5{suffix}', val_stats['patches_below_0_0_5' if not is_swa else 'patches_below_0_0_5_swa'], epoch_1000x)
            log_writer.add_scalar(f'val/patches_below_0_0_1{suffix}', val_stats['patches_below_0_0_1' if not is_swa else 'patches_below_0_0_1_swa'], epoch_1000x)

    # wandb logging
    if (args.use_wandb):
        log_dict = {f"val/val_loss{suffix}": val_stats['loss' if not is_swa else 'loss_swa']}
        epoch_dict = {f"val/epoch_1000x": epoch_1000x} if not is_swa else {} # epoch only needs to be logged once

        log_dict.update(epoch_dict)

        # In the "train_and_validate_one_epoch function":
        #       1) while swa hasn't started yet:
        #           -> we only log normal model stats. Because of this: commit = [True if not epoch_end else False, True].
        #       2) when swa starts
        #           -> we will call the "log_stats" function first to log normal stats then swa stats.
        #           -> Then, if we are logging normal stats:
        #               - commit = [False, False]
        #           -> If we are logging swa_stats:
        #               - commit = [True if not epoch_end else False, True].

        if (not args.do_swa):
            commit = [True if not epoch_end else False, True]
        else:
            commit = [False, False] if not is_swa else [True if not epoch_end else False, True]

        wandb.log(
            log_dict,
            commit=commit[0]
        )

        if (epoch_end):
            wandb.log(
                {
                f"val/patches_below_0_3{suffix}": val_stats['patches_below_0_3' if not is_swa else 'patches_below_0_3_swa'],
                f"val/patches_below_0_1{suffix}": val_stats['patches_below_0_1' if not is_swa else 'patches_below_0_1_swa'],
                f"val/patches_below_0_0_5{suffix}": val_stats['patches_below_0_0_5' if not is_swa else 'patches_below_0_0_5_swa'],
                f"val/patches_below_0_0_1{suffix}": val_stats['patches_below_0_0_1' if not is_swa else 'patches_below_0_0_1_swa'],
                }, 
                commit=commit[1]
            )
