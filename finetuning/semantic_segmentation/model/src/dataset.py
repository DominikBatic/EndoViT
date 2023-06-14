import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from pathlib import Path
import torch
import sys

segmentation_dir_path = "./finetuning/semantic_segmentation/model"
sys.path.append(segmentation_dir_path)
from src.custom_transforms import custom_RandomResizedCrop, custom_Resize, custom_RandomHorizontalFlip, custom_Normalize, custom_ToTensor

########################################################################################################################
############################################# CholecSeg8k Dataset Class ################################################
########################################################################################################################
# CholecSeg8k description:

# STRUCTURE

#   CholecSeg8k
#   ├── video01 # 1st level
#   │   ├── video01_00080 # 2nd level
#   │   │   ├── frame_80_endo.png                        # raw_image
#   │   │   ├── frame_80_endo_watershed_mask.png         # ground truth semantic segmentation
#   │   │   ├── frame_80_endo_mask.png                   # annotation mask
#   │   │   ├── frame_80_endo_color_mask.png             # color mask
#   │   │   ├── ...
#   │   │   ├──
#   │   │   ├── frame_159_endo.png
#   │   │   ├── frame_159_endo_watershed_mask.png
#   │   │   ├── frame_159_endo_mask.png
#   │   │   └── frame_159_endo_color_mask.png
#   │   ├── ...
#   │   └── video01_00080
#   ├── ...
#   └── video55
#
#
# The dataset has 2 level directory structure:
#   -> 1st level: 
#       - Represents Cholec80 videos (CholecSeg8k contains 17 out of 80 Cholec80 videos)
#   -> 2nd level:
#       - Each directory is a video clip of 80 consequtive frames from the corresponding video.
#       - The naming convention is {video_name}_{starting_frame_number}, 
#         e.g. if the name is video01_00080, then this directory 
#         contains frames 80 - 159 from video01 of Cholec80 dataset.
#
#
# NOTE:
#   The dataset contains 13 classes in total, including class 0 - background.
#
#   For each frame, 4 separate .png files are saved.
#   1) raw image (e.g. frame_80_endo.png)
#   2) ground truth semantic segmentation mask <=> watershed_mask (e.g. frame_80_endo_watershed_mask.png)
#   3) annotation mask (e.g. frame_80_endo_mask.png)
#       - This mask was used by the annotation tool during the annotation process and can be ignored.
#   4) color mask (e.g. frame_80_endo_color_mask.png)
#       - This mask is equivallent to the watershed mask but color values for each class are different.
#       - In watershed masks each class was assigned 1 color, but Red, Green and Blue values of each of
#         these colors are the same. (e.g. class 0 = color [127, 127, 127]).
#       - On the other hand, in color masks each class was also assigned 1 color, but there is no
#         restriction on RGB values. (e.g. class 0 = color [])
#       - The color mask is used for visualization purposes.


class CholecSeg8k(Dataset):
    def __init__(self, data_dir, RP_file, dataset_type="train", transform=None, vids_to_take=None, verbose=True):
        """
        @params:
            -> data_dir (str):
                - path to the root directory of CholecSeg8k dataset
            -> RP_file (str):
                - path to the .txt file containing relative paths from "data_dir" to each individual image of CholecSeg8k dataset
                - each row of this file will have 5 comma-separated paths:
                    1) relative path to the input image
                    2) relative path to the corresponding ground truth semantic segmentation mask (which we created in the pre-processing)
                    3) relative path to the corresponding watershed mask
                    4) relative path to the corresponding annotation mask (this mask was used by annotation tool during annotation and is ignored)
                    5) relative path to the corresponding color mask (this mask is for visualization purposes)
            -> dataset_type (str, default: "train"):
                - specification which dataset it is, options: [train / val / test]
            -> transform (callable, default: None):
                - transformation to be applied to each image and the corresponding ground truth
            -> vids_to_take(List[int], default: None)
                - An optional list of CholecSeg8k videos to use for the dataset. If set to None all videos are taken.
        """

        assert dataset_type in ["train", "val", "test"], f"When building CholecSeg8k dataset, argument \"dataset_type\" should be one of the following [train, val, test], received \"dataset_type={dataset_type}\" instead."

        self.data_dir = Path(data_dir)
        self.RP_file = RP_file
        assert os.path.exists(RP_file), "Please provide a path to an existing .csv file."
        self.relative_paths = pd.read_csv(RP_file)
        assert self.relative_paths.shape == (8080, 5), f"Expected \"Relative Paths\" to be of shape (8080, 5), but received the shape {self.relative_paths.shape}"

        if (transform is None):
            self.transform = self.get_default_transform(dataset_type)
        else:
            self.transform = transform

        self.all_CholecSeg8k_videos = [1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37, 43, 48, 52, 55]

        # We follow the splits as given in the following paper:
        #   -> "Analysis of Current Deep Learning Networks for Semantic Segmentation of Anatomical Structures in Laparoscopic Surgery"
        self.CholecSeg8k_splits = {
            "train": [1, 9, 18, 20, 24, 25, 26, 28, 35, 37, 43, 48, 55],
            "val": [17, 52],
            "test": [12, 27],
        }

        if (vids_to_take is None):
            self.vids_to_take = self.CholecSeg8k_splits[dataset_type]
        else:
            valid_videos = []
            invalid_videos = []

            for vid in vids_to_take:
                if (vid in self.all_CholecSeg8k_videos):
                    valid_videos.append(vid)
                else:
                    invalid_videos.append(vid)

            self.vids_to_take = valid_videos
            assert self.vids_to_take, "Invalid videos!"
            if (invalid_videos):
                print("WARNING: Some videos you wished to take aren't present in CholecSeg8k dataset:")
                print(f"\t -> Invalid videos: {invalid_videos}")

            self.relative_paths = self.relative_paths.loc[self.relative_paths["image"].apply(lambda x: int(x.split("/")[0][-2:]) in self.vids_to_take)]
        
        left_side = f"{dataset_type.capitalize()} videos:"
        if (verbose):
            print(f"{left_side:>13s} {self.vids_to_take}")

    def get_default_transform(self, dataset_type="train"):

        ########################## Hyperparameters ##########################
        # resize image to this size
        input_size   = 224 
        # mean and std are taken from the full Cholec80 dataset
        dataset_mean = [0.3464, 0.2280, 0.2228]
        dataset_std  = [0.2520, 0.2128, 0.2093]

        # probability of performing random horizontal flip
        flip_probability = 0.5

        # scale and ratio parameters of RandomResizedCrop
        RRC_scale = [0.2, 1.0]
        RRC_ratio = [3.0 / 4.0, 4.0 / 3.0]

        ######################## END_Hyperparameters ########################

        transform = None

        if (dataset_type == "train"):
            transform = T.Compose([
                        custom_RandomResizedCrop(input_size, RRC_scale, RRC_ratio),
                        custom_RandomHorizontalFlip(flip_probability),
                        custom_ToTensor(),
                        custom_Normalize(dataset_mean, dataset_std)
                    ])
        
        elif (dataset_type == "val" or dataset_type == "test"):
            transform = T.Compose([
                        custom_Resize((input_size, input_size)),
                        custom_ToTensor(),
                        custom_Normalize(dataset_mean, dataset_std)
                    ])
        else:
            assert False, f"In function \"CholecSeg8k.get_default_transform\" argument \"dataset_type\" should be one of the following [train, val, test], received \"dataset_type={dataset_type}\" instead."

        return transform

    def __len__(self):
        return len(self.relative_paths)
    
    def __getitem__(self, index):
        relative_path       = self.relative_paths.iloc[index, :]

        img_path             = self.data_dir / relative_path[0]
        ground_truth_path    = self.data_dir / relative_path[1]
        watershed_mask_path  = self.data_dir / relative_path[2] # watershed mask is not used
        annotation_mask_path = self.data_dir / relative_path[3] # annotation mask is not used
        color_mask_path      = self.data_dir / relative_path[4] # color mask is not used

        image               = Image.open(img_path)
        ground_truth        = Image.open(ground_truth_path)

        sample = {"image": image, "mask": ground_truth}

        if (self.transform is not None):
            sample          = self.transform(sample)

        return sample