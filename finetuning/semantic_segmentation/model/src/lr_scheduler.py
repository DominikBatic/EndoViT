# Implementation taken from github repo: "https://github.com/facebookresearch/mae".

import math

class CosineScheduler(object):
    def __init__(self, config, optimizer):
        self.optimizer = optimizer

        self.total_epochs = config["General Hyperparams"]["epochs"]
        self.warmup_epochs = config["General Hyperparams"]["warmup_epochs"]
        self.base_lr = config["General Hyperparams"]["base_lr"]
        self.min_lr = config["General Hyperparams"]["min_lr"]

    def step(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs 
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
        
        # track min and max LR while updating
        min_lr = float("inf")
        max_lr = float("-inf")

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"] # min(lr * param_group["lr_scale"], self.min_lr)
            else:
                param_group["lr"] = lr

            min_lr = min(param_group["lr"], min_lr)
            max_lr = max(param_group["lr"], max_lr)

        return {"min_lr": min_lr, "max_lr": max_lr}
    
    def __str__(self):
        to_print = ""
        to_print += f"CosineScheduler:\n"
        to_print += f"   -> Decays the learning rate with half-cycle cosine after warmup.\n"
        to_print += f"   -> NOTE: LR will be updated at the start of each training step,\n"
        to_print += f"            not at the end of each epoch.\n\n"
        to_print += f"   Warmup Epochs: {self.warmup_epochs}\n"
        to_print += f"   Min allowed LR: {self.min_lr:g}"

        return to_print
    
    # This Scheduler has no State Dictionary
    def load_state_dict(self, state_dict):
        pass