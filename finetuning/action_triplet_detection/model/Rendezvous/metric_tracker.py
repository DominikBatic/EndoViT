import torch
import numpy as np
from collections import defaultdict, deque

# get accuracy as a metric
    #@args:
    #   preds (torch.Tensor, shape: [B, C]), where B = batch_size, C = #classes 
    #       - predictions of the model
    #   targets (torch.Tensor, shape: [B, C])
    #       - ground truth labels
    #
    #@output
    #   acc (float)
    #       - accuracy of the model
def accuracy_element_wise(preds, targets):
    p_flat = preds.ravel()
    t_flat = targets.ravel()

    acc = (p_flat == t_flat).sum() / len(t_flat)

    return float(acc.item())

def accuracy_prediction_wise(preds, targets):
    acc = torch.all(preds==targets, dim=1).sum() / len(targets)

    return float(acc.item())

class AccuracyTracker(object):
    def __init__(self, delimiter='\t'):
        # this list will store element/prediction-wise accuracy for each previous video
        self.per_video_metrics = []
        
        # element-wise accuracy for current video 
        self.element_wise = {
            "correct" : 0,
            "total_count" : 0
        }
        # prediction-wise accuracy for current video 
        self.prediction_wise = {
            "correct" : 0,
            "total_count" : 0
        }

        self.delimiter = delimiter

    def update(self, preds, targets):
        self.update_accuracy_element_wise(preds, targets)
        self.update_accuracy_prediction_wise(preds, targets)

    def update_accuracy_element_wise(self, preds, targets):
        p_flat = preds.ravel()
        t_flat = targets.ravel()

        self.element_wise["correct"] += int((p_flat == t_flat).sum().item())
        self.element_wise["total_count"] += len(t_flat)

    def update_accuracy_prediction_wise(self, preds, targets):
        if (targets.dim() == 1):
            preds = torch.unsqueeze(preds, 0)
            targets = torch.unsqueeze(targets, 0)
        
        count = len(targets) if targets.dim() > 1 else 1

        self.prediction_wise["correct"] += int(torch.all(preds==targets, dim=1, keepdim=True).sum().item())
        self.prediction_wise["total_count"] += count

    def get_accuracy(self):
        return {
            "element_wise_acc": float(self.element_wise["correct"] / self.element_wise["total_count"]),
            "prediction_wise_acc" : float(self.prediction_wise["correct"] / self.prediction_wise["total_count"])
        }
    
    def get_mean_acc_per_video(self):
        video_count = len(self.per_video_metrics)

        if (video_count > 0):
            mean_element_wise_acc = sum([video_metrics["element_wise_acc"] for video_metrics in self.per_video_metrics]) / video_count
            mean_prediction_wise_acc = sum([video_metrics["prediction_wise_acc"] for video_metrics in self.per_video_metrics]) / video_count

        else:
            mean_element_wise_acc = -1.
            mean_prediction_wise_acc = -1.

        return {
            "mean_element_wise_acc": mean_element_wise_acc,
            "mean_prediction_wise_acc": mean_prediction_wise_acc
        }

    def __str__(self):
        acc_str = []
        for name, meter in self.get_mean_acc_per_video().items():
            acc_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(acc_str)
    
    # resets metrics for the current video only
    def reset(self):
        self.element_wise["correct"] = 0
        self.element_wise["total_count"] = 0
        self.prediction_wise["correct"] = 0
        self.prediction_wise["total_count"] = 0

    # resets current video metrics as well as stored previous video metrics
    def global_reset(self):
        self.reset()
        self.per_video_metrics = []

    def video_end(self):
        self.per_video_metrics.append(self.get_accuracy())
        self.reset()

        
# code taken from MAE repository (https://github.com/facebookresearch/mae)
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

# code taken from MAE repository (https://github.com/facebookresearch/mae)
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter