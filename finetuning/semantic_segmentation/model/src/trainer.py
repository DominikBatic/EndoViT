from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import src.util as util
import datetime
import time
import math
import subprocess

import os, sys
segmentation_dir_path = "./finetuning/semantic_segmentation/model"
sys.path.append(segmentation_dir_path)

import torch.cuda.amp as amp

import src.dataset as datasets
import src.metrics as MyMetrics
import src.lr_scheduler as LR_Scheduler
import src.custom_transforms as cT
import src.metric_logger  as ml
import src.amp_scaler as Scaler
import src.losses as MyLosses
import torchvision.transforms as T
from DPT.dpt.models import DPTSegmentationModel


# TODO: (Optional) Iteration Accumulation, (Optional) Distributed Training
class Trainer(object):
    def __init__(self, config):
        self.config = config

        util.header("Initializing Trainer")

        ############################### Setup ###############################

        self.output_dir = Path(self.config["Project Setup"]["output_dir"])
        self.log_dir = self.output_dir / self.config["Project Setup"]["log_dir"]
        self.log_txt_file = self.log_dir / self.config["Project Setup"]["log_file"]
        self.save_ckpt_format = self.config["Project Setup"]["checkpoint_format"]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(self.config["General Hyperparams"]["device"] if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.total_epochs = self.config["General Hyperparams"]["epochs"]
        self.do_optimizations = self.config["General Hyperparams"]["performance_optimizations"]

        self.use_wandb = self.config["Loggers"]["WandB"]["enable"]
        self.log_every_n_steps = self.config["Loggers"]["log_every_n_steps"]

        self.ignore_index = self.config["General Hyperparams"]["ignore_index"] or -100

        ########################## Resume Training ##########################

        self.checkpoint = self.load_pretrained_checkpoint(config)

        self.start_epoch = self.checkpoint.get("epoch") or 0
        self.model_state_dict = self.checkpoint.get("model")
        self.optimizer_state_dict = self.checkpoint.get("optimizer")
        # Default CosineScheduler doesn't have a state_dict so this will always return None.
        self.scheduler_state_dict = self.checkpoint.get("scheduler")

        # Load scaler_state_dict if and only if performance_optimizations are enabled.
        self.scaler_state_dict = self.checkpoint.get("scaler") if self.do_optimizations else None

        ########################### Initialization ##########################
        
        # Init dataloaders.
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._init_dataloaders(self.config)

        # Init the model.
        self.model = self._init_model(self.config, state_dict=self.model_state_dict)
        self.model.to(self.device)

        # Init the loss.
        self.loss_fn = self._init_loss(self.config)
        self.loss_fn.to(self.device) # For nn.CrossEntropyLoss this has no effect.

        # Init metrics.
        metrics = self._init_metrics(self.config)
        self.train_metrics, self.val_metrics, self.test_metrics = [m.to(self.device) for m in metrics]

        # Init the optimizer.
        self.optimizer = self._init_optimizer(self.config, self.model, state_dict=self.optimizer_state_dict)

        # Init the scheduler.
        self.scheduler = self._init_scheduler(self.config, self.optimizer, state_dict=self.scheduler_state_dict)

        # Init the scaler.
        self.scaler = self._init_scaler(self.config, state_dict=self.scaler_state_dict) if self.do_optimizations else None

    
    def save_best_model(self, best_result_dict):
        self.extension = ".pth"

        metric, value = best_result_dict["metric"][0], best_result_dict["metric"][1]
        epoch = best_result_dict["epoch"]

        to_save = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
        }
        if (self.scaler is not None):
            to_save["scaler"] = self.scaler.state_dict()

        # To avoid creating many fails, all checkpoints will initially be saved as output_dir / "best_ckpt.pth".
        # After the training is done the "best_ckpt.pth" will be renamed as the "checkpoint_format" given
        # in the .config file.
        output_path = self.output_dir / ("best_ckpt" + self.extension)

        torch.save(to_save, str(output_path))

        util.new_section()
        print(f"Best checkpoint saved! Epoch: {epoch} {metric}: {value:.4f}")

        # Create the path corresponding to "checkpoint_format" so that we can rename "best_ckpt.pth" at the end of training.
        metric_str = f"{metric}_{value:.4f}"
        epoch_str = f"Epoch_{epoch}"
        output_path = output_path.parent / (self.save_ckpt_format.format(metric=metric_str, epoch=epoch_str) + self.extension)

        best_result_dict["ckpt_path"] = str(output_path)

        return
    
    
    @torch.no_grad()
    def _evaluate(self,
        model,
        dataloader,
        loss_fn,
        metrics,
        is_test=False,
        epoch=0,
        best_result_dict={},
        ):

        model.eval()

        stats = {}
        total_loss = 0.

        # If logging to wandb, we will visualize few of the resulting segmentations.
        # This determines how many images per epoch will be plotted.
        #----------------------------------
        if (not is_test):
            max_n_results_to_plot = 3
            size = len(dataloader) # num_batches
            n_results_to_plot = max_n_results_to_plot if size >= max_n_results_to_plot else 1
            offset = int(0.2 * size) # we skip this many batches at the beginning and end
            if (size <= 2*offset):
                offset = 0
            step = int((size - 2*offset) / n_results_to_plot)

            # We plot 1 image in each of the following batches.
            at_which_batches_to_plot = list(range(offset, n_results_to_plot * step, step))
        #----------------------------------
        else:
            what_to_plot = {
                "oblig_plots": {
                    "6":  0,
                    "8": 59,
                    "16": 12,
                },
                "aux_plots": {
                    "0": 21,
                    "1": 46,
                    "2":  4,
                    "3": 34,
                    "4":  7,
                    "5": 31,
                    "7": 40,
                    "9": 34,
                    "10": 55,
                    "11": 17,
                    "12": 30,
                    "13": 13,
                    "14": 17,
                    "15":  8,
                }
            }


        prefix = "test" if is_test else "val"

        header = 'Validation at Epoch: [{}]'.format(epoch) if not is_test else "Testing"
        print_freq = 10
        metric_logger = ml.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', ml.SmoothedValue(window_size=print_freq, fmt="{median:.4f} ({global_avg:.4f})"))
        metric_logger.add_meter('IoU',  ml.SmoothedValue(window_size=print_freq, fmt="{value:.2f} ({global_avg:.2f})"))
        metric_logger.add_meter('dice', ml.SmoothedValue(window_size=print_freq, fmt="{value:.2f} ({global_avg:.2f})"))
        metric_logger.add_meter('acc',  ml.SmoothedValue(window_size=print_freq, fmt="{value:.2f} ({global_avg:.2f})"))

        for current_step, sample in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
            inputs, targets = sample["image"], sample["mask"]

            inputs = inputs.to(self.device, non_blocking=True)
            # Target segmentation masks are saved as 3 channel images, we only need 1 channel.
            targets = targets[:, 0, :, :].to(self.device, non_blocking=True)

            if (self.do_optimizations and self.device == torch.device("cuda")):
                with amp.autocast():
                    logits = model(inputs)
                    loss = loss_fn(logits, targets) if self.loss_fn_to_use != "CombinedLoss" else loss_fn(logits, targets, epoch)
            else:
                logits = model(inputs)
                loss = loss_fn(logits, targets) if self.loss_fn_to_use != "CombinedLoss" else loss_fn(logits, targets, epoch)

            loss_value = loss.item()
            total_loss += loss_value * inputs.shape[0] # batch_size
            metric_logger.update(loss=loss_value)

            # batch_metrics are not used
            preds = torch.argmax(logits, dim=1)
            batch_metrics = metrics(preds, targets)
            # computes the global average over all seen images
            current_step_metrics = metrics.compute()

            metric_logger.update(IoU =current_step_metrics['IoU_scores'].sum().item() / current_step_metrics["Observed_classes"].sum().item())
            metric_logger.update(dice=current_step_metrics['Dice_scores'].sum().item() / current_step_metrics["Observed_classes"].sum().item())
            metric_logger.update(acc =current_step_metrics['Acc_PerPixel'].item())

            # Visualize some results to WandB
            if (self.use_wandb):
                if (not is_test):
                    if (current_step in at_which_batches_to_plot):
                        inputs = inputs.detach().cpu()
                        targets = targets.detach().cpu()
                        preds = preds.detach().cpu()

                        wandb.log(
                            {
                            f"{prefix}/Example_{str(current_step)}": wandb.Image(util.create_wandb_plots(self.config, inputs[0], targets[0], preds[0]))
                            }
                            , commit=True)
                else:
                    if (what_to_plot["oblig_plots"].get(str(current_step)) is not None or what_to_plot["aux_plots"].get(str(current_step)) is not None):
                        inputs = inputs.detach().cpu()
                        targets = targets.detach().cpu()
                        preds = preds.detach().cpu()

                        if (what_to_plot["oblig_plots"].get(str(current_step)) is not None):
                            wandb.log(
                                {
                                f"Obligatory_{prefix}/Example_{str(current_step)}": wandb.Image(util.create_wandb_plots(self.config, inputs[what_to_plot["oblig_plots"][str(current_step)]], targets[what_to_plot["oblig_plots"][str(current_step)]], preds[what_to_plot["oblig_plots"][str(current_step)]]))
                                }
                                , commit=True)
                        if (what_to_plot["aux_plots"].get(str(current_step)) is not None):
                            wandb.log(
                                {
                                f"Auxiliary_{prefix}/Example_{str(current_step)}": wandb.Image(util.create_wandb_plots(self.config, inputs[what_to_plot["aux_plots"][str(current_step)]], targets[what_to_plot["aux_plots"][str(current_step)]], preds[what_to_plot["aux_plots"][str(current_step)]]))
                                }
                                , commit=True)
        # -------------------------------------------

        end_metrics = metrics.compute()
        stats.update({
            "Loss": total_loss / len(dataloader.dataset),
            "IoU": end_metrics['IoU_scores'].sum().item() / end_metrics["Observed_classes"].sum().item(),
            "Dice": end_metrics['Dice_scores'].sum().item() / end_metrics["Observed_classes"].sum().item(),
            "Acc": end_metrics['Acc_PerPixel'].item(),
            })
        
        # Log to WandB
        if (self.use_wandb):
            wandb.log(
                {
                f"{prefix}/loss": total_loss / len(dataloader.dataset),
                f"{prefix}/IoU": end_metrics["IoU_scores"].sum().item() / end_metrics["Observed_classes"].sum().item(),
                f"{prefix}/Dice": end_metrics["Dice_scores"].sum().item() / end_metrics["Observed_classes"].sum().item(),
                f"{prefix}/Acc": end_metrics["Acc_PerPixel"].item(),
                f"{prefix}/Classes": end_metrics["Observed_classes"].sum().item(),
                }, 
                commit=True,
            )


        # -------------------------------------------
        # Save best model
        # -------------------------------------------
        mean_IoU = end_metrics["IoU_scores"].mean().item()
        
        if (not is_test):
            if (best_result_dict["metric"][1] < mean_IoU):
                best_result_dict["metric"][1] = mean_IoU
                best_result_dict["epoch"] = epoch

                self.save_best_model(best_result_dict)

        return stats


    def _train_one_epoch(self,
        epoch,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        loss_fn,
        train_metrics,
        scaler,
        ):
        
        model.train(True)

        general_stats = {}
        train_stats = {}

        total_loss = 0.
        total_imgs = 0

        # Metrics are tracked and logged in 3 different ways:
        # 1) Printing to std_out:
        #    -> For this we use MAE's MetricLogger.
        #    -> Metric Logger will track loss during training and print out it's median and average every 10 batches.
        # 2) WandB logging (if enabled)
        #    -> Logs the loss and all metrics at the end of each training step.
        #    -> NOTE: These are the metrics from "src/metrics.py". By default IoU and Dice scores are meaned per image (not batch),
        #             while pixel Accuracy counts correct pixels over all seen images and divides by the total number of
        #             pixels. Because of this the logged metrics will be running averages.
        # 3) Writing to log.txt file:
        #    -> Everything saved in general_stats or train_stats dictionary gets logged to log_dir/log.txt file
        #       at the end of every epoch. E.g. Total loss will be averaged over all training images and only logged at
        #       the end of the epoch.


        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10   
        metric_logger = ml.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', ml.SmoothedValue(window_size=print_freq, fmt="{median:.4f} ({global_avg:.4f})"))

        for current_step, sample in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
            inputs, targets = sample["image"], sample["mask"]

            optimizer.zero_grad()

            # Learning rates are updated every step, not at the end of every epoch.
            current_LR = scheduler.step(current_step / len(train_dataloader) + epoch)

            general_stats.update(current_LR)

            inputs = inputs.to(self.device, non_blocking=True)
            # Target segmentation masks are saved as 3 channel images, we only need 1 channel.
            targets = targets[:, 0, :, :].to(self.device, non_blocking=True)

            if (self.do_optimizations and self.device == torch.device("cuda")):
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = loss_fn(logits, targets) if self.loss_fn_to_use != "CombinedLoss" else loss_fn(logits, targets, epoch)
            else:
                logits = model(inputs)
                loss = loss_fn(logits, targets) if self.loss_fn_to_use != "CombinedLoss" else loss_fn(logits, targets, epoch)

            loss_value = loss.item()
            assert math.isfinite(loss_value), f"Loss is {loss_value}. Stopping training ... "
            metric_logger.update(loss=loss_value)
            total_loss += loss_value * inputs.shape[0] # batch_size
            total_imgs += inputs.shape[0]

            # batch_metrics are not used
            preds = torch.argmax(logits, dim=1)
            batch_metrics = train_metrics(preds, targets)
            # computes the global average over all seen images
            current_step_metrics = train_metrics.compute()

            if (self.do_optimizations and self.device == torch.device("cuda")):
                scaler(loss, optimizer)
            else:
                loss.backward()
                optimizer.step()

            # Log to WandB
            if (self.use_wandb and (current_step + 1) % self.log_every_n_steps == 0):
                epoch_1000x = int((current_step / len(train_dataloader) + epoch) * 1000)

                wandb.log(
                    {
                    "train/train_loss": total_loss / total_imgs,
                    "train/epoch_1000x": epoch_1000x,
                    "train/max_lr": general_stats["max_lr"],
                    "train/min_lr": general_stats["min_lr"],
                    "train/Observed_classes": current_step_metrics["Observed_classes"].sum().item(),
                    "train/IoU": current_step_metrics["IoU_scores"].sum().item() / current_step_metrics["Observed_classes"].sum().item(),
                    "train/Dice": current_step_metrics["Dice_scores"].sum().item() / current_step_metrics["Observed_classes"].sum().item(),
                    "train/Acc": current_step_metrics["Acc_PerPixel"].item(),
                    }, 
                    commit=True,
                )

        # ---------------- Epoch END ----------------

        epoch_metrics = train_metrics.compute()
        train_stats.update({
            "Loss": total_loss / len(train_dataloader.dataset),
            "IoU": epoch_metrics['IoU_scores'].sum().item() / epoch_metrics["Observed_classes"].sum().item(),
            "Dice": epoch_metrics['Dice_scores'].sum().item() / epoch_metrics["Observed_classes"].sum().item(),
            "Acc": epoch_metrics['Acc_PerPixel'].item(),
            })

        return (general_stats, train_stats)


    def run_training(self):
        print(f"Starting training ...")
        print(f"\t -> Total epochs: {self.total_epochs}")
        print(f"\t -> Start epoch: {self.start_epoch}")

        start_time = time.time()

        best_result_dict = {
            "metric": ["mIoU", float("-inf")],
            "epoch" : -1,
            "ckpt_path": "",
        }

        for epoch in range(self.start_epoch, self.total_epochs):
            # Both train and validation stats should be dictionaries of losses/metrics and other statistics
            # you want to log.
            #
            #   e.g. {"CrossEntropyLoss" : CE_value, "IoU": IoU_value, "Dice": dice_value, "Acc": acc_value}
            #
            # General stats should also be a dictionary of the same form.
            #
            #   e.g. {"min_lr": min_lr_value, "max_lr", max_lr_value}
            
            util.header(f"Epoch {epoch}")
            general_stats, train_stats = self._train_one_epoch(
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_dataloader,
                self.loss_fn,
                self.train_metrics,
                self.scaler,
            )

            util.new_line(times=2)
            # best_model_saving is performed in the _evaluate function
            val_stats = self._evaluate(
                self.model,
                self.val_dataloader,
                self.loss_fn,
                self.val_metrics,
                is_test=False,
                epoch=epoch,
                best_result_dict=best_result_dict,
            )

            # reset metrics
            self.train_metrics.reset()
            self.val_metrics.reset()

            log_stats = {
                'epoch': epoch,
                **general_stats,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'n_parameters': self.pytorch_train_params
                }

            with open(self.log_txt_file, mode="a", encoding="utf-8") as f:
                to_write = ""
                to_write += "{"

                for i, (k, v) in enumerate(log_stats.items()):
                    if (k == "epoch"):
                        to_write += f"\"{k}\": {v:2d}"
                    elif (k == "n_parameters"):
                        to_write += f"\"{k}\": {v / 1e+6:3.2f} (M)"
                    else:
                        to_write += f"\"{k}\": {v:g}"
                    to_write += ", " if i+1 != len(log_stats.items()) else "}\n"

                f.write(to_write)


        # Rename the "best_ckpt.pth" checkpoint.
        src = self.output_dir / ("best_ckpt" + self.extension)
        dest = best_result_dict["ckpt_path"]
        subprocess.run(["mv", f"{src}", f"{dest}"])

        util.header("Training Done!", separator="#")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        return best_result_dict


    def run_testing(self, checkpoint_path):

        # this load function comes from DPT base_model
        self.model.load(checkpoint_path)
        self.model.to(self.device)

        util.new_section("Testing", separator="#")

        test_stats = self._evaluate(
                self.model,
                self.test_dataloader,
                self.loss_fn,
                self.test_metrics,
                is_test=True,
            )
        
        # reset metrics
        self.test_metrics.reset()
        
        util.header(title="Final Results")

        to_write = ""

        for k, v in test_stats.items():
            to_write += f"\"{k}\": {v:g}\n"

        print(to_write)

        return


    def load_pretrained_checkpoint(self, config):
        util.header(title="Loading Checkpoint", separator="*")

        # Resume training from checkpoint.
        self.checkpoint = {}
        self.resume_from_checkpoint = self.config["General Hyperparams"]["resume_training"]
        self.load_checkpoint = self.config["General Hyperparams"]["load_checkpoint"]

        # There are two options:
        #   1) resume_from_checkpoint
        #       -> Will continue the training from where it left off.
        #       -> Last epoch, model_state_dict, optimizer_state_dict,
        #          scheduler_state_dict will be loaded. If performance
        #          optimizations are enabled, scaler_state_dict will
        #          also be loaded.
        #
        #   2) load_checkpoint
        #       -> Will start training from scratch, but will load a
        #          pretrained model_state_dict.
        #
        # NOTE: Resume_from_checkpoint has priority over load_checkpoint

        if (self.resume_from_checkpoint):
            self.checkpoint = torch.load(self.resume_from_checkpoint)

        elif(self.load_checkpoint):
            self.checkpoint = torch.load(self.load_checkpoint)

            self.checkpoint["epoch"] = 0
            self.checkpoint["optimizer"] = None
            self.checkpoint["scheduler"] = None
            self.checkpoint["scaler"] = None

        # If loading from a pretrained checkpoint, then mae_ckpt should not be loaded.
        if (self.checkpoint):
            print(f"Checkpoint loaded from: {self.resume_from_checkpoint if self.resume_from_checkpoint else self.load_checkpoint}")
            print(f"\t -> Setting \"mae_ckpt\" = \"\".")
            self.config["Model"]["MAE Hyperparams"]["mae_ckpt"] = ""
        else:
            print("No checkpoint loaded!")

        return  self.checkpoint


    def _init_transforms(self, config):
        util.header(title="Transforms", separator="*")

        ########################## Hyperparameters ##########################

        # resize image to this size
        self.input_size = config["Dataset"]["Transformations"]["img_size"] or 224
        # Default mean and std are taken from the full Cholec80 dataset
        dataset_mean = config["Dataset"]["Transformations"]["dataset_mean"] or [0.3464, 0.2280, 0.2228]
        dataset_std  = config["Dataset"]["Transformations"]["dataset_std"] or [0.2520, 0.2128, 0.2093]

        # probability of performing random horizontal flip
        flip_probability = config["Dataset"]["Transformations"]["flip_probability"] or 0.5

        # scale and ratio parameters of RandomResizedCrop
        RRC_scale = config["Dataset"]["Transformations"]["RandomResizedCrop_scale"] or [0.2, 1.0]
        RRC_ratio = [3.0 / 4.0, 4.0 / 3.0]

        ############################ Overfitting ############################

        # If overfitting is enabled we don't wish to use random transformations.
        overfitting_enabled = config["Dataset"]["Overfitting"]["enabled"]

        if (overfitting_enabled):
            config["Dataset"]["Transformations"]["train"] = ["Resize", "ToTensor", "Normalize"]
            config["Dataset"]["Transformations"]["val"]   = ["Resize", "ToTensor", "Normalize"]
            config["Dataset"]["Transformations"]["test"]  = ["Resize", "ToTensor", "Normalize"]

            print("Overfitting enabled!")
            print(f"\t -> Train transform = Val transform = Test transform = {config['Dataset']['Transformations']['train']}", end="\n\n")
            
        ######################## Transform Dictionary #######################

        transform_dictionary = {
            "RandomResizedCrop": cT.custom_RandomResizedCrop(self.input_size, RRC_scale, RRC_ratio),
            "RandomHorizontalFlip": cT.custom_RandomHorizontalFlip(flip_probability),
            "Resize": cT.custom_Resize(self.input_size),
            "ToTensor": cT.custom_ToTensor(),
            "Normalize": cT.custom_Normalize(dataset_mean, dataset_std),
        }
        
        ########################## Build Tranforms ##########################

        self.train_transform = T.Compose([
            transform_dictionary[transform] for transform in config["Dataset"]["Transformations"]["train"]
            ])
        
        self.val_transform = T.Compose([
            transform_dictionary[transform] for transform in config["Dataset"]["Transformations"]["val"]
            ])
        
        self.test_transform = T.Compose([
            transform_dictionary[transform] for transform in config["Dataset"]["Transformations"]["test"]
            ])
        
        print("Train transform:\n", self.train_transform, end="\n\n")
        print("Val transform:\n", self.val_transform, end="\n\n")
        print("Test transform:\n", self.test_transform, end="\n\n")

        return (self.train_transform, self.val_transform, self.test_transform)


    def _init_datasets(self, config, train_transform, val_transform, test_transform):
        util.header(title="Datasets", separator="*")

        ############################### Setup ###############################

        self.data_dir = config["Dataset"]["Initialization"]["dataset_dir"]
        self.RP_file_path = config["Dataset"]["Initialization"]["RP_file"]
        self.splits = config["Dataset"]["Splits"]

        ############################ Overfitting ############################

        overfitting_enabled = config["Dataset"]["Overfitting"]["enabled"]
        overfit_n_images = config["Dataset"]["Overfitting"]["num_images"]

        if (overfitting_enabled):
            self.train_dataset = datasets.CholecSeg8k(
                self.data_dir,
                self.RP_file_path,
                dataset_type="train",
                transform=train_transform,
                vids_to_take=self.splits["train_videos"],
                verbose=False,
            )

            # equally distribute samples
            size = len(self.train_dataset)
            assert overfit_n_images <= size, "Can't overfit to {overfit_n_images} images, the training dataset has only {size} images. Please make sure the number of overfit images is lower than the train dataset size." 
            step = int(size / overfit_n_images)
            
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(0, overfit_n_images * step, step)))
            self.val_dataset = self.train_dataset
            self.test_dataset = self.train_dataset

            print("Overfitting enabled ...")
            print(f"\t -> Overfitting to {overfit_n_images} images from the train dataset.")
            print( "\t -> Train dataset = Validation dataset = Test dataset")

            return (self.train_dataset, self.val_dataset, self.test_dataset)

        ###################### Data Efficient Training ######################

        DET_enabled = config["Dataset"]["Data Efficient Training"]["enabled"]
        DET_train_videos = config["Dataset"]["Data Efficient Training"]["train_videos"]

        if (DET_enabled):
            for video in DET_train_videos:
                if (video not in self.splits["train_videos"]):
                    raise ValueError(f"Video{video:2d} isn't in the train dataset.")
            self.splits["train_videos"] = DET_train_videos

            print("Data Efficient Training enabled ...")
            util.new_section(new_line=False)

        ########################### Build Datasets ##########################

        self.train_dataset = datasets.CholecSeg8k(
            self.data_dir,
            self.RP_file_path,
            dataset_type="train",
            transform=train_transform,
            vids_to_take=self.splits["train_videos"],
            verbose=False,
            )
        
        self.val_dataset = datasets.CholecSeg8k(
            self.data_dir,
            self.RP_file_path,
            dataset_type="val",
            transform=val_transform,
            vids_to_take=self.splits["val_videos"],
            verbose=False,
            )
        
        self.test_dataset = datasets.CholecSeg8k(
            self.data_dir,
            self.RP_file_path,
            dataset_type="test",
            transform=test_transform,
            vids_to_take=self.splits["test_videos"],
            verbose=False,
            )
        
        print(f"Train videos: {self.splits['train_videos']}")
        print(f"  Val videos: {self.splits['val_videos']}")
        print(f" Test videos: {self.splits['test_videos']}")

        util.new_section()

        print(f"Total train #images: {len(self.train_dataset)}")
        print(f"  Total val #images: {len(self.val_dataset)}")
        print(f" Total test #images: {len(self.test_dataset)}")
        
        return (self.train_dataset, self.val_dataset, self.test_dataset)


    def _init_dataloaders(self, config):
        #util.header(title="Dataloaders", separator="*")

        ########################## Build Transforms #########################

        self.train_transform, self.val_transform, self.test_transform = self._init_transforms(config)
        transforms = [self.train_transform, self.val_transform, self.test_transform]

        ########################### Build Datasets ##########################

        self.train_dataset, self.val_dataset, self.test_dataset = self._init_datasets(config, *transforms)

        ######################### Build Dataloaders #########################
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config["General Hyperparams"]["batch_size"],
            num_workers=config["General Hyperparams"]["num_workers"],
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            )
        
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config["General Hyperparams"]["batch_size"],
            num_workers=config["General Hyperparams"]["num_workers"],
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            )
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config["General Hyperparams"]["batch_size"],
            num_workers=config["General Hyperparams"]["num_workers"],
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            )
        
        #print("Train dataloader:\n", self.train_dataloader, end="\n\n")
        #print("Val dataloader:\n", self.val_dataloader, end="\n\n")
        #print("Test dataloader:\n", self.test_dataloader, end="\n\n")
        
        return (self.train_dataloader, self.val_dataloader, self.test_dataloader)


    def _init_model(self, config, state_dict=None):
        util.header(title="Model", separator="*")

        ############################### Setup ###############################

        DPT_hyperparams = config["Model"]["DPT Hyperparams"]
        MAE_hyperparams = config["Model"]["MAE Hyperparams"]

        DPT_kwargs = {
            "readout": DPT_hyperparams["readout"],
            "features": DPT_hyperparams["features"],
            "use_bn": DPT_hyperparams["use_bn"],
        }

        ############################# Load Model ############################

        self.model= DPTSegmentationModel(
            DPT_hyperparams["num_classes"],
            path = None, # We don't wish to load any saved checkpoints here. Checkpoint loading is done in the trainer.
            backbone = DPT_hyperparams["backbone"],
            mae_hyperparams=MAE_hyperparams,
            # unnecessary hyperparams, because these are set by default
            **DPT_kwargs
        )

        print("", end="\n\n")
        util.header(title="Full Model", total_length=50)

        print(self.model)

        util.header(title="Number of Parameters", total_length=50)
        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.pytorch_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print('number of params (M): %.2f' % (self.pytorch_total_params / 1.e6))
        print('number of trainable params (M): %.2f' % (self.pytorch_train_params / 1.e6))
        
        if (state_dict):
            util.new_section()
            self.model.load_state_dict(state_dict)
            print("Loaded model state dict!")

        return self.model


    def _init_loss(self, config):
        util.header(title="Loss", separator="*")

        ############################### Setup ###############################

        self.loss_fn_to_use = config["General Hyperparams"]["loss"]

        ########################## Loss Dictionary ##########################

        loss_dictionary = {
            "CrossEntropy": nn.CrossEntropyLoss(ignore_index=self.ignore_index),
            "FocalLoss": MyLosses.FocalLoss(ignore_index=self.ignore_index),
            "CombinedLoss": MyLosses.CombinedLoss(ignore_index=self.ignore_index)
        }

        ############################# Build loss ############################

        self.loss_fn = loss_dictionary.get(self.loss_fn_to_use)
        assert self.loss_fn is not None, print(f"Invalid loss function. Supported loss functions: {list(loss_dictionary.keys())}.")

        print(f"Loss Function: {self.loss_fn}")

        # When using Focal Loss the last conv layer needs to be initialized properly. Otherwise, the Focal Loss doesn't work properly at the start of the training.
        if (self.loss_fn_to_use == "FocalLoss"):
            torch.nn.init.normal_(self.model.scratch.output_conv[4].weight, mean=0., std=0.0001)
            torch.nn.init.constant_(self.model.scratch.output_conv[4].bias, -2)

            print("\t -> Initialized the last Conv2d layer of the model as:")
            print("\t\t Weight: Normal(mean=0, std=0.0001)")
            print("\t\t   Bias: -2")

        return self.loss_fn


    def _init_metrics(self, config):
        util.header(title="Metrics", separator="*")

        ############################### Setup ###############################

        metrics_to_use = config["General Hyperparams"]["metrics"]
        num_classes = config["Model"]["DPT Hyperparams"]["num_classes"]

        ######################### Metric Dictionary #########################

        metrics_dictionary = {
            "IoU+Dice+PerPixelAcc": MyMetrics.SegmentationMetrics(num_classes, ignore_index=self.ignore_index),
        }

        ########################### Build Metrics ###########################

        self.train_metrics = metrics_dictionary.get(metrics_to_use)
        assert self.train_metrics is not None, print(f"Invalid metric. Supported metrics: {list(metrics_dictionary.keys())}.")

        self.val_metrics = metrics_dictionary[metrics_to_use]
        self.test_metrics = metrics_dictionary[metrics_to_use]

        print(self.train_metrics)

        return (self.train_metrics, self.val_metrics, self.test_metrics)


    def _init_optimizer(self, config, model, state_dict=None):
        util.header(title="Optimizer", separator="*")

        ############################### Setup ###############################

        optimizer_to_use = config["General Hyperparams"]["optimizer"]
        base_lr = config["General Hyperparams"]["base_lr"]
        base_wd = config["General Hyperparams"]["base_wd"]
        
        ####################### Build Parameter Groups ######################

        backbone_param_groups = model.param_groups
        backbone_parameters = set(sum([pg["params"] for pg in backbone_param_groups], []))

        decoder_parameters = set(model.parameters()) - backbone_parameters
        decoder_param_group = [{ 
            "params": list(decoder_parameters),
            "lr": base_lr,
            "weight_decay": base_wd,
        },]

        opt_parameters = []
        opt_parameters.extend(backbone_param_groups)
        opt_parameters.extend(decoder_param_group)
        
        ######################## Optimizer Dictionary #######################

        optimizer_dictionary = {
            "AdamW": optim.Adam(opt_parameters, lr=base_lr, weight_decay=base_wd),
            "SGD": optim.SGD(opt_parameters, lr=base_lr, weight_decay=base_wd, momentum=0.9)
        }
        
        ########################## Build Optimizer ##########################

        self.optimizer = optimizer_dictionary.get(optimizer_to_use)
        assert self.optimizer is not None, print(f"Invalid optimizer. Supported optimizers: {list(optimizer_dictionary.keys())}.")

        print(f"Optimizer: {optimizer_to_use}")
        print(f"Base LR: {base_lr:g}")
        print(f"Base Weight Decay: {base_wd:g}")

        print("")
        util.header(title="Optimizer Parameter Groups", total_length=50)

        print(self.optimizer)

        if (state_dict):
            util.new_section()
            self.optimizer.load_state_dict(state_dict)
            print("Loaded optimizer state dict!")

        return self.optimizer


    def _init_scheduler(self, config, optimizer, state_dict=None):
        util.header(title="Scheduler", separator="*")

        ############################### Setup ###############################

        scheduler_to_use = config["General Hyperparams"]["scheduler"]

        ######################## Scheduler Dictionary #######################

        scheduler_dictionary = {
            "cosine": LR_Scheduler.CosineScheduler(config, optimizer),
        }

        ########################## Build Scheduler ##########################
        
        self.scheduler = scheduler_dictionary.get(scheduler_to_use)
        assert self.scheduler is not None, print(f"Invalid scheduler. Supported schedulers: {list(scheduler_dictionary.keys())}.")

        print(self.scheduler)

        if (state_dict):
            util.new_section()
            self.scheduler.load_state_dict(state_dict)
            print("Loaded scheduler state dict!")

        return self.scheduler
    
    def _init_scaler(self, config, state_dict=None):
        util.header(title="Scaler", separator="*")

        self.scaler = Scaler.AmpScaler()

        print("Using torch.cuda.amp.GradScaler ...")

        if (state_dict):
            util.new_section()
            self.scaler.load_state_dict(state_dict)
            print("Loaded scaler state dict!")

        return self.scaler