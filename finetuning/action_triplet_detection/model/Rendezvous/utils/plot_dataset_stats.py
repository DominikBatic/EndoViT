import os
import sys
import torch
import argparse
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from matplotlib.lines import Line2D

#################################################################################################
####################################### Helper Functions ########################################
#################################################################################################

colors_dict = {
    "orange": "#ff8200",
    "red": "#ff4040",
    "blue": "#0a75ad",
    "green": "#3ac63d",
    "black": "#000000",
    "pink": "#fe6aef",
    "yellow": "#ffd700",
    "brown": "#6c3f2d",
    "purple": "#7c3fa8",
    "cian": "#26e9e9"
}

# get labels to print
def get_label_map(class_occurrences, labels, type="middle"):
    calculated_max = np.amax(class_occurrences, axis=0)
    calculated_argmax = np.argmax(class_occurrences, axis=0)

    class_density_of_occurrences = class_occurrences / np.expand_dims(class_occurrences.sum(axis=1), axis=1)
    calculated_density_max = np.amax(class_density_of_occurrences, axis=0)
    calculated_density_argmax = np.argmax(class_density_of_occurrences, axis=0)

    occurrences = {
        "class_occurrences": class_occurrences,
        "class_density_of_occurrences": class_density_of_occurrences
    }

    statistics = {
        "max": calculated_max,
        "argmax": calculated_argmax,
        "density_max": calculated_density_max,
        "density_argmax": calculated_density_argmax
    }

    return _form_label_map(occurrences, statistics, labels, type=type)

def _form_label_map(occurrences, statistics, labels, type="middle"):
    assert statistics["max"].shape[0] == len(labels), "Error number of labels doesn't match the counts."
    label_map = {
        "train": [],
        "val": [],
        "test": []
    }

    if (type=="max"):
        for i, index in enumerate(statistics["argmax"]):
            if (index == 0):
                label_map["train"].append(str(labels[i])) if statistics["max"][i] != 0 else label_map["train"].append("")
                label_map["val"].append("")
                label_map["test"].append("")
            elif (index == 1):
                label_map["train"].append("")
                label_map["val"].append(str(labels[i])) if statistics["max"][i] != 0 else label_map["val"].append("")
                label_map["test"].append("")
            elif (index == 2):
                label_map["train"].append("")
                label_map["val"].append("")
                label_map["test"].append(str(labels[i])) if statistics["max"][i] != 0 else label_map["test"].append("")
            else:
                assert False, "Error invalid index!"

        return (label_map, None)
        
    elif (type=="middle"):
        for i, l in enumerate(labels):
            label_map["train"].append("")
            label_map["val"].append(str(l)) if (occurrences["class_occurrences"][2][i] != 0) else label_map["val"].append("")
            label_map["test"].append("")

        return (label_map, statistics)

    return (label_map, None)

# plot statistics histogram of a dataloader
# this version plots histograms for train, val and test dataset overlayed
def histogram(class_occurrences, kfold):
    train_counts = class_occurrences[0]
    val_counts = class_occurrences[1]
    test_counts = class_occurrences[2]

    bins = list(range(train_counts.shape[0] + 1))
    label_map, _ = get_label_map(class_occurrences, bins[:-1], type="max")

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.cla()

    N_train, bins_train, patches_train  = ax.hist(bins[:-1], bins, weights=train_counts, density=True, alpha = 0.5)
    N_val, bins_nval, patches_val       = ax.hist(bins[:-1], bins, weights=val_counts, density=True, alpha = 0.5)
    N_test, bins_test, patches_test     = ax.hist(bins[:-1], bins, weights=test_counts, density=True, alpha = 0.5)

    ax.bar_label(patches_train, labels=label_map["train"], fontsize=11)
    ax.bar_label(patches_val, labels=label_map["val"], fontsize=11)
    ax.bar_label(patches_test, labels=label_map["test"], fontsize=11)

    custom_lines = [Line2D([0], [0], color=colors_dict["orange"], lw=4),
                    Line2D([0], [0], color=colors_dict["blue"], lw=4),
                    Line2D([0], [0], color=colors_dict["green"], lw=4)]
    
    ax.legend(custom_lines, ["train", "val", "test"], prop={"size": 20})
    ax.set_title(f"Test Fold {kfold} Dataset Class Occurences", fontsize=20, pad=5)
    ax.set_ylabel("Count Density", fontsize=20)
    ax.set_xlabel("Triplet Class", fontsize=20)

    return fig

# this version plots histograms for train, val and test dataset next to each other
def histogram_v2(class_occurrences, kfold):
    bins = list(range(class_occurrences[0].shape[0] + 1))
    labels = np.vstack([np.asarray(bins[:-1])] * 3).transpose(1, 0)
    label_map, statistics = get_label_map(class_occurrences, labels[:, 0], type="middle")

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.cla()

    colors = ["red", "blue", "lime"]
    legend_labels = ["Train", "Val", "Test"]
    counts = class_occurrences.transpose(1, 0)

    N, bins_combined, patches = ax.hist(labels, bins, rwidth=0.6, weights=counts, density=True, histtype='bar', color=colors, label=legend_labels, alpha=1.)

    if statistics is not None:
        for i, (rect, label) in enumerate(zip(patches[1], label_map["val"])):
            height = (patches[statistics["density_argmax"][i]][i]).get_height() #rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
            )

    plt.rcParams["legend.loc"] = "upper left"
    ax.legend(prop={'size': 20})
    ax.set_title(f"Test Fold {kfold} Dataset Class Occurences", fontsize=20, pad=5)
    ax.set_ylabel("Count Density", fontsize=20)
    ax.yaxis.set_label_coords(-0.01, 0.5)
    ax.set_xlabel("Triplet Class", fontsize=20)

    fig.tight_layout()
    return fig

#################################################################################################
################################## Setting Up Argument Parser ###################################
#################################################################################################

#%% @args parsing
parser = argparse.ArgumentParser()
# data
parser.add_argument('--data_dir', type=str, default='/path/to/dataset', help='path to dataset?')
parser.add_argument('--output_dir', type=str, default='/path/to/output_dir', 
                    help=
                    """
                    Path where the output file containing per video class counts will be saved.
                    Per video statistics will be saved to "per_video_counts.txt" and statistics
                    for different test folds to "per_test_fold_counts.txt".
                    """)

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

FLAGS, unparsed = parser.parse_known_args()

#################################################################################################
######################################## Initialization #########################################
#################################################################################################

data_dir            = FLAGS.data_dir
output_dir          = FLAGS.output_dir
use_wandb           = FLAGS.use_wandb

# output file path
output_file_path_vc       = os.path.join(output_dir, 'per_video_counts.txt')
output_file_path_vc_hr    = os.path.join(output_dir, 'per_video_counts_human_redable.txt')
output_file_path_tfc      = os.path.join(output_dir, 'per_test_fold_counts.txt')
output_file_path_tfc_hr   = os.path.join(output_dir, 'per_test_fold_counts_human_redable.txt')

#%% data loading : variant and split selection (Note: original paper used different augumentation per epoch)

# wandb logging
if (not use_wandb):
    os.environ["WANDB_MODE"]="offline"

# log in
wandb.login()

date_str = datetime.now().strftime("%H:%M-%d.%m.%y_")

# initialize the project
wandb.init(
    project="Dataset Statistics",
    name=f"Histograms {date_str}",
    tags=["statistics"]
)
          
#################################################################################################
####################################### Per Video Counts ########################################
#################################################################################################

print("\nFetching per video counts ...\n")

videos = [1, 2, 4, 5, 6, 8, 10, 12, 13, 14, 15, 18, 22, 23, 25, 26, 27, 29, 31, 32, 35, 36, 40, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 60, 62, 65, 66, 68, 70, 73, 74, 75, 78, 79, 80]
video_records = ['VID{}'.format(str(v).zfill(2)) for v in videos]

per_video_counts = {}

for i, video in enumerate(video_records):
    triplet_file = os.path.join(data_dir, 'triplet', '{}.txt'.format(video))
    triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',',)[:, 1:]

    counts = triplet_labels.sum(axis=0)
    assert counts.shape == (100,), f"ERROR: Expected count shape (100,) got {counts.shape} for video '{video}'."
    per_video_counts[videos[i]] = counts

    print(f"Processed Video {i+1:2d}/{len(videos)}")

print("\nSaving results to {output_file_path} ...")

with open(output_file_path_vc, 'w') as f:
    with open(output_file_path_vc_hr, 'w') as f_hr:
        labels = list(range(100))
        header = "".join([" "] * 9) + "|" +  "|".join([f"  {l:2d}  " for l in labels])
        separator = "".join(["-"] * len(header))

        print(f"{header}", file=f_hr)
        print(f"{separator}", file=f_hr)

        for i in range(len(videos)):
            count_list      = list(per_video_counts[videos[i]])
            count_string    = "".join([f" {count}" for count in count_list])
            print(f"{str(videos[i])}{count_string}", file=f)
            
            count_string_hr = "".join([f" | {count:4d}" for count in count_list])
            print(f"Video {videos[i]:2d}{count_string_hr}", file=f_hr)

#################################################################################################
##################################### Per Test Fold Counts ######################################
#################################################################################################

print("\nCalculating statistics for different choices of the test fold ...")
cholect45_crossval = {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
            }

fold_counts_map = {}
    
for kfold in range(1, 6):
    train_videos = sum([v for k,v in cholect45_crossval.items() if k!=kfold], [])
    test_videos  = sum([v for k,v in cholect45_crossval.items() if k==kfold], [])

    val_videos   = train_videos[-5:]
    train_videos = train_videos[:-5]

    train_count = np.zeros(100, dtype=int)
    val_count = np.zeros(100, dtype=int)
    test_count = np.zeros(100, dtype=int)

    for video in train_videos:
        train_count += per_video_counts[video]

    for video in val_videos:
        val_count   += per_video_counts[video]

    for video in test_videos:
        test_count  += per_video_counts[video]

    fold_counts_map[kfold] = {"train": train_count, "val": val_count, "test": test_count}

#################################################################################################
###################################### Plotting Histograms ######################################
#################################################################################################
    if (use_wandb):

        count_map = np.vstack([train_count, val_count, test_count])

        wandb.log({"Histogram": wandb.Image(histogram_v2(count_map, kfold))})

    print(f"Processed Fold {kfold}/5")

#################################################################################################
################################### Save Per Test Fold Counts ###################################
#################################################################################################
with open(output_file_path_tfc, 'w') as f:
    with open(output_file_path_tfc_hr, 'w') as f_hr:
        labels = list(range(100))
        header = "".join([" "] * 6) + "|" + "|".join([f"   {l:2d}  " for l in labels])
        separator = "".join(["-"] * len(header))

        print(f"{header}", file=f_hr)
        print(f"{separator}", file=f_hr)

        for kfold in range(1, 6):
            train_total         = fold_counts_map[kfold]["train"].sum()
            val_total           = fold_counts_map[kfold]["val"].sum()
            test_total          = fold_counts_map[kfold]["test"].sum()

            train_count_list    = list(fold_counts_map[kfold]["train"])
            val_count_list      = list(fold_counts_map[kfold]["val"])
            test_count_list     = list(fold_counts_map[kfold]["test"])

            train_count_string  = "".join([f" {count}" for count in train_count_list])
            val_count_string    = "".join([f" {count}" for count in val_count_list])
            test_count_string   = "".join([f" {count}" for count in test_count_list])

            print(f"{kfold}{train_count_string}{val_count_string}{test_count_string}", file=f)

            train_count_string_hr   = "".join([f" | {count:5d}" for count in train_count_list])
            val_count_string_hr     = "".join([f" | {count:5d}" for count in val_count_list])
            test_count_string_hr    = "".join([f" | {count:5d}" for count in test_count_list])

            print(f"Fold{kfold:2d}:   Total_Train: {train_total:6d}   #   Total_Val: {val_total:6d}   #   Total_Test: {test_total:6d}", file=f_hr)
            print(f"{separator}", file=f_hr)
            print(f"{'Train':5s}{train_count_string_hr}", file=f_hr)
            print(f"{  'Val':5s}{val_count_string_hr  }", file=f_hr)
            print(f"{ 'Test':5s}{test_count_string_hr }", file=f_hr)
            print(f"{separator}", file=f_hr)

wandb.finish()

print("")
print("".join(["#"] * 50))
print("\nDone!")