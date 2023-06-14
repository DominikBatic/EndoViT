import torch
from torchmetrics import Metric
from torchmetrics.functional import dice, jaccard_index
from torchmetrics.functional.classification import multiclass_confusion_matrix

# Not used.
class PerPixelAccuracy(Metric):
    def __init__(self, num_classes, average='micro', multidim_average='global', ignore_index=None):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        full_state_update = False

    def update(self, preds, targets):
        assert preds.shape == targets.shape

        self.correct += torch.sum(preds == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


# Not used.
class mDice(Metric):
    def __init__(self, num_classes, average=None, mdmc_average=None, ignore_index=None):
        super().__init__()
        self.add_state("mDice", default=torch.tensor([0] * num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.dice = lambda preds, targets: dice(preds, targets, num_classes=num_classes, average=average, mdmc_average=mdmc_average, ignore_index=ignore_index)

        full_state_update = False

    def update(self, preds, targets):

        batch_size = len(targets)

        self.mDice += sum(self.dice(p.view(-1), t.view(-1)) for p, t in zip(preds, targets))
        self.num_samples += torch.tensor(batch_size)

    def compute(self):
        return self.mDice / self.num_samples


# Not used.
class mIoU(Metric):
    def __init__(self, num_classes, average=None, ignore_index=None):
        super().__init__()
        self.add_state("mIoU", default=torch.tensor([0] * num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.IoU = lambda preds, targets: jaccard_index(preds, targets, num_classes, average=average, ignore_index=ignore_index, task='multiclass')

        full_state_update = False

    def update(self, preds, targets):

        batch_size = len(targets)

        self.mIoU += sum(self.IoU(p, t) for p, t in zip(preds, targets))
        self.num_samples += torch.tensor(batch_size)

    def compute(self):
        return self.mIoU / self.num_samples
    

class SegmentationMetrics(Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.add_state("IoU_total", default=torch.zeros(num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("dice_total", default=torch.zeros(num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct_pixels", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        self.confusion_matrix = lambda preds, targets: multiclass_confusion_matrix(preds, targets, num_classes, ignore_index=ignore_index)
        self.num_classes = num_classes

        full_state_update = True

    def update(self, preds, targets):
        assert preds.shape == targets.shape

        batch_size = len(targets)

        preds = preds.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)

        cms = [self.confusion_matrix(p, t) for p, t in zip(preds, targets)]

        Intersections = torch.stack([cm.diagonal() for cm in cms])
        DiceUnions = torch.stack([cm.sum(dim=0) + cm.sum(dim=1) for cm in cms])

        assert Intersections.shape == (batch_size, self.num_classes)
        assert DiceUnions.shape == (batch_size, self.num_classes)

        IoU_scores = (Intersections.to(torch.float32) / (DiceUnions - Intersections + 1e-8)).sum(dim=0)
        Dice_scores = (2*Intersections.to(torch.float32) / (DiceUnions + 1e-8)).sum(dim=0)
        CorrectPixels = Intersections.sum()
        TotalPixels = (DiceUnions.sum() / 2).to(torch.long)
        NumImagesPerClass = torch.where(DiceUnions > 0, 1, 0).to(torch.long).sum(dim=0)

        assert NumImagesPerClass.shape == (self.num_classes,)

        self.IoU_total += IoU_scores
        self.dice_total += Dice_scores
        self.num_images += NumImagesPerClass
        self.correct_pixels += CorrectPixels
        self.total_pixels += TotalPixels

    def compute(self):
        observed_classes = torch.where(self.num_images > 0, 1, 0).to(torch.long)
        return {"IoU_scores": self.IoU_total.to(torch.float32) / (self.num_images + 1e-8), "Dice_scores": self.dice_total.to(torch.float32) / (self.num_images + 1e-8), "Acc_PerPixel": self.correct_pixels.to(torch.float32) / self.total_pixels, "Observed_classes": observed_classes}

    def __str__(self):
        to_print = ""
        to_print += "SegmentationMetrics:\n"
        to_print += "    -> IoU scores (per class)\n"
        to_print += "    -> Dice scores (per class)\n"
        to_print += "    -> per pixel Accuracy\n\n"
        to_print += "    NOTE: IoU scores are computed for each class on each image separately.\n"
        to_print += "          The final IoU scores of each class are returned by averaging\n"
        to_print += "          over all images. The same is done with Dice scores.\n"
        to_print += "          On the other hand, Accuracy is computed by summing correctly\n"
        to_print += "          predicted pixels accross all images and dividing by the total\n"
        to_print += "          number of pixels."

        return to_print
    

class SegmentationMetricsV2(Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.add_state("confusion_matrix", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")

        self.calculate_confusion_matrix = lambda preds, targets: multiclass_confusion_matrix(preds, targets, num_classes, ignore_index=ignore_index)
        self.num_classes = num_classes

        full_state_update = True

    def update(self, preds, targets):
        assert preds.shape == targets.shape

        preds = preds.flatten()
        targets = targets.flatten()

        cm = self.calculate_confusion_matrix(preds, targets)
        
        assert cm.shape == (self.num_classes, self.num_classes)
        self.confusion_matrix += cm

        return
    
    def compute(self):
        Intersections = self.confusion_matrix.diagonal()
        DiceUnions = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1)

        assert Intersections.shape == (self.num_classes,)
        assert DiceUnions.shape == (self.num_classes,)

        IoU_scores = Intersections.to(torch.float32) / (DiceUnions - Intersections + 1e-8)
        Dice_scores = 2*Intersections.to(torch.float32) / (DiceUnions + 1e-8)
        CorrectPixels = Intersections.sum()
        TotalPixels = (DiceUnions.sum() / 2)

        ObservedClasses = self.confusion_matrix.sum(dim=1)
        ObservedClasses = torch.where(ObservedClasses > 0, 1, 0).to(torch.long)

        assert IoU_scores.shape == (self.num_classes,)
        assert Dice_scores.shape == (self.num_classes,)
        assert ObservedClasses.shape == (self.num_classes,)

        return {"IoU_scores": IoU_scores, "Dice_scores": Dice_scores, "Acc_PerPixel": CorrectPixels.to(torch.float32) / TotalPixels, "Observed_classes": ObservedClasses}

    def __str__(self):
        to_print = ""
        to_print += "SegmentationMetrics:\n"
        to_print += "    -> IoU scores (per class)\n"
        to_print += "    -> Dice scores (per class)\n"
        to_print += "    -> per pixel Accuracy\n\n"
        to_print += "    NOTE: IoU scores are computed for each class accross all images.\n"
        to_print += "          The same is done with Dice scores. Accuracy is computed by\n"
        to_print += "          summing correctly predicted pixels accross all images and\n"
        to_print += "          dividing by the total number of pixels."

        return to_print