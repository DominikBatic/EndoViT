import subprocess
import os
#import cv2
import glob
import random
from pathlib import Path
import gdown

import argparse

import synapseclient
import synapseutils

###########################################################################################################
##############################################  CHEAT SHEAT  ##############################################
###########################################################################################################

# 1) Download and Pre-process Endo700k
#    download_datasets.py --all --synapse_email YOUR_EMAIL --synapse_password YOUR_PASSWORD
#    
#    -> Add:
#       --download_output_dir YOUR_DOWNLOAD_PATH to specify where to download the datasets.
#    -> Add:
#       --prepare_output_dir YOUR_PREPROCESS_PATH to specify where to save the pre-processed datasets.
#    -> By default --download_output_dir == --prepare_output_dir == ./datasets/Endo700k
#    -> If --download_output_dir == --prepare_output_dir the pre-processing is done in-place and all
#       unnecessary files are deleted.
#    -> To download HeiCo a synapse account is required: https://www.synapse.org/
#    -> After creating it, pass the synapse email and password as the above arguments
#    -> 700 GB required for downloading (HeiCo is large).
#    -> If pre-processing is done in-place, Endo700k is only around 150 GB.
#
# 2) Download and Pre-process specific datasets
#    download_dataset.py [LIST OF DATASETS]
#
#    Allowed dataset tags:
#    1) --ESAD
#    2) --LapGyn4
#    3) --SurgicalActions160
#    4) --GLENDA
#    5) --hSDB-instrument
#    6) --HeiCo_dataset (if you put this tag, then pass the --synapse_email and --synapse_password as well)
#    7) --PSI_AVA
#    8) --DSAD
#
# 3) Only download the datasets
#    download_dataset.py [LIST OF DATASETS or --all] --mode d
#
#    -> By default datasets are saved at: ./datasets/Endo700k
#    -> Add:
#       --download_output_dir YOUR_DOWNLOAD_PATH to specify where to download the datasets.
#
# 4) Only pre-process the datasets
#    download_dataset.py [LIST OF DATASETS or --all] --mode p
#
#    -> Assumes the datasets are already downloaded at --download_output_dir.
#    -> Only pre-processes the datasets and saves the resulting output at --prepare_output_dir.
#    -> Add:
#       --download_output_dir YOUR_DOWNLOAD_PATH to specify where to find the downloaded datasets.
#    -> Add:
#       --prepare_output_dir YOUR_PREPROCESS_PATH to specify where to save the processed datasets.

###########################################################################################################
############################################  USE INSTRUCTIONS  ###########################################
###########################################################################################################

# In our project we used 9 datasets (Endo700k collection) to pre-train a ViT model (EndoViT):
# 1) ESAD
# 2) LapGyn4
# 3) SurgicalActions160
# 4) GLENDA
# 5) hSDB-instrument
# 6) HeiCo
# 7) PSI-AVA
# 8) DSAD
# 9) Cholec80
#
# Afterwards, we evaluated EndoViT as a feature extraction backbone on 3 downstream tasks:
# 1) Semantic Segmentation on CholecSeg8k dataset
# 2) Action Triplet Detection on CholecT45 dataset
# 3) Surgical Phase Recognition on Cholec80 dataset
#
# This script is meant for downloading and pre-processing of the datasets 1-8 in Endo700k collection.
# For the other datasets, i.e. Cholec80, CholecT45 and CholecSeg8k we have written separate instruction
# notes on how to download them.
#
# This script allows you to download the datasets at --download_output_dir path. Afterwards, the datasets
# can be pre-processed and the resulting output saved at --prepare_output_dir. The pre-processing
# removes all unnecessary files, such as dataset annotations and only keeps the raw images that will be
# used for the pre-training of EndoViT. This results in an ImageFolder structure as follows:
#
# prepare_output_dir
# ├── ESAD
# ├── LapGyn4
# ├── SurgicalActions160
# ├── GLENDA
# ├── hSDB-instrument
# ├── HeiCo_dataset
# ├── PSI-AVA
# └── DSAD
#     ├── img_1.png
#     ├── ...
#     └── img_n.png
#
#
# where each dataset is saved under its own subdirectory and contains only the extracted raw images.
#
# If --download_output_dir and --prepare_output_dir are the same path, the pre-processing is done in-place,
# i.e. the extra files are deleted. On the other hand, if they specify different paths, the original datasets
# will not be changed. However, this requires significantly more memory. (Instead of 150 GB you should allocate
# approximately 1.5 TB. The largest individual dataset is HeiCo with approximately 600 GB. Because of this, even
# if the final pre-processed images have only 150 GB, you will need at least 700 GB to download the datasets.)

# IMPORTANT NOTE: In majority of the cases we download the full datasets with all of their corresponding files.
#                 However, sometimes the datasets contained data which was not relevant for our project. For
#                 example, hSDB-instrument's Laparoscopic Cholecystectomy videos come from Cholec80. Consequently,
#                 we don't download those files. If your goal is to use this script to download the original
#                 datasets for your own use, please check if all of the files are downloaded in the code bellow.
#
# IMPORTANT NOTE 2: Sometimes when downloading from google drive we experienced problems. In particular with
#                   hSDB-instrument dataset. Often the download will finish without downloading the
#                   full .zip file. If this occurs an error will be reported while unzipping. The only
#                   solution that worked so far was to re-start the download until it successfully finishes.
#                   (we used both gdown and wget: wget got us better results)
#                   If at --download_output_dir there already exists a subfolder with the dataset's name,
#                   this script will skip the download of that dataset. So to re-start, just delete the
#                   hSDB-instrument subfolder and re-run the script.

# Below is a list of all the arguments that can be passed to download_datasets.py script.

def get_argument_parser():
    parser = argparse.ArgumentParser('Download And Prepare Datasets', add_help=True)

    parser.add_argument('-d', '--download_output_dir', default='./datasets/Endo700k',
                        help='Path where to download the datasets.')
    
    parser.add_argument('-p', '--prepare_output_dir', default='./datasets/Endo700k',
                        help=
                        """
                        After downloading, the datasets can be pre-processed if appropriate flag is raised.
                        The pre-processed datasets will be saved at this path's location.

                        Note: If download and prepare_output_dir are the same, the original file structure
                              will not be preserved. For more information look at --mode flag.
                        """)
    
    parser.add_argument('-a', '--all', action='store_true', 
                        help='Download/Prepare all datasets.')
    
    parser.add_argument('--mode', type=str, default="dnp", choices=["dnp", "d", "p"],
                        help=
                        """
                        There are 3 possible modes:

                        1) dnp (default) = download and prepare datasets
                            -> All selected datasets will be downloaded into --download_output_dir. Then
                               the datasets will be pre-processed for EndoViT pre-training. The resulting files
                               will be saved at --prepare_output_dir.
                            -> The datasets are pre-processed by mostly extracting frames from video
                               sequences at 1 FPS and afterwards saved into an ImageFolder structure.
                               I.e. for each processed dataset we will create an output subdirectory
                               containing all the extracted images for that dataset.
                               E.g. if we have two datasets the resulting structure will be:
                               
                               prepare_output_dir
                               ├── Dataset_1
                               |   ├── img_1.png
                               |   ├── ...
                               |   └── img_n.png
                               └── Dataset_2
                                   ├── img_1.png
                                   ├── ...
                                   └── img_n.png

                            -> If download and prepare_output_dir are the same, then the preparation will 
                               be done "in-place", i.e. the original file structure will be destroyed.


                        2) d = download only
                            -> All selected datasets will be downloaded into --download_output_dir.

                        3) p = prepare only
                            -> This assumes the datasets have already been downloaded at
                               --download_output_dir, and the original dataset structure is unchanged.
                            -> If download and prepare_output_dir are the same, then the preparation will 
                               be done "in-place", i.e. the original file structure will be destroyed. 
                        """)

    parser.add_argument('--ESAD', action='store_true',
                        help='Download/Prepare ESAD dataset.')

    parser.add_argument('--LapGyn4', action='store_true',
                        help='Download/Prepare LapGyn4 dataset.')

    parser.add_argument('--SurgicalActions160', action='store_true',
                        help='Download/Prepare SurgicalActions160 dataset.')

    parser.add_argument('--GLENDA', action='store_true',
                        help='Download/Prepare GLENDA dataset.')

    parser.add_argument('--hSDB_instrument', action='store_true',
                        help='Download/Prepare hSDB-instrument dataset.')

    parser.add_argument('--HeiCo_dataset', action='store_true',
                        help=
                        """
                        Download/Prepare HeiCo dataset.

                        Note: In order to download HeiCo dataset a synapse account is needed (https://www.synapse.org/).
                              After making the account, pass the corresponding email and password as --synapse_email 
                              and --synapse_password flags.

                              We follow the instructions listed at: https://www.synapse.org/#!Synapse:syn21903917/wiki/601992
                              And using their synapse script download the **COMPLETE dataset**.

                              Your email and password are only forwarded to the synapse client in order to log in. For 
                              more information look at "download_HeiCo" function.

                        Note 2: This dataset is very large (~ 600 GB).
                        """)
    
    parser.add_argument('--synapse_email', type=str, default='',
                        help='Email of your synapse account. Only needed if you wish to download HeiCo dataset.')

    parser.add_argument('--synapse_password', type=str, default='',
                        help='Password of your synapse account. Only needed if you wish to download HeiCo dataset.')   

    parser.add_argument('--PSI_AVA', action='store_true',
                        help=
                        """
                        Download/Prepare PSI-AVA dataset.

                        Note: This dataset is large. If you don't need the annotations and TAPIR models, but only raw frames
                              set the flag --PSI_AVA_unannotated_frames_only.
                        """)
    
    parser.add_argument('--PSI_AVA_download_all', action='store_true',
                            help=
                            """
                            Setting this flag will download the whole dataset instead of just the raw frames that were relevant
                            for our project.
                            """)
    
    parser.add_argument('--DSAD', action='store_true',
                            help='Download/Prepare Dresden Surgical Anatomy Dataset (DSAD) dataset.')

    return parser


# Before going into download/prepare functions, first take a look at the main function at the bottom of this script.
def download_ESAD(args):
    """
    STRUCTURE:
        -> Frame naming convention: RARP{procedure_number}_frame_{frame_number(in order of occurence)}.jpg
                
        ESAD
        ├── train
        |   ├── obj.names
        |   |
        |   ├── set1
        |   |   ├── RARP2_frame_1.jpg
        |   |   ├── RARP2_frame_1.txt
        |   |   ├── RARP2_frame_2.jpg
        |   |   ├── RARP2_frame_2.jpg
        |   |   ...
        |   |   ├── RARP2_frame_9501.jpg
        |   |   └── RARP2_frame_9501.txt
        |   |
        |   └── set2
        |       ├── RARP4_frame_1.jpg
        |       ├── RARP4_frame_1.txt
        |       ...
        |       ├── RARP4_frame_31016.jpg
        |       └── RARP4_frame_31016.txt
        |
        ├── val
        |   ├── obj.names
        |   |
        |   └── obj
        |       ├── RARP1_frame_1.txt
        |       ├── RARP1_frame_2.jpg
        |       ...
        |       ├── RARP1_frame_7713.jpg
        |       └── RARP1_frame_7822.txt
        |
        ├── test
        |   ├── RARP3_frame_2.jpg
        |   ├── RARP3_frame_3.jpg
        |   ...
        |   ├── RARP3_frame_7687.jpg
        |   └── RARP3_frame_7688.jpg
        |
        └── test_labels
            ├── RARP3_frame_2.txt
            ├── RARP3_frame_3.txt
            ...
            ├── RARP3_frame_7687.txt
            └── RARP3_frame_7688.txt

        Note:
            - Train, val, test, test_labels subfolders have their own separate download links. We
              group them together into one ESAD folder.

    Download links:
        - train
            "https://drive.google.com/u/0/uc?id=1CnYAzZRVEDGK1TycGBb8SnMgyvzeZrie"
        - val
            "https://drive.google.com/u/0/uc?id=17rWwuWKFZFxQ0DTRs5cmUzU2Vb5PScol" 
        - test
            "https://drive.google.com/u/0/uc?id=1gho-oGzUbNgnZmBZ2GDKWWOcs1VI-z0O" 
        - test_labels
            "https://drive.google.com/u/0/uc?id=16srrq1NIso1mI2YKtHMIPyn5bZbcCyo3"


    DESCRIPTION:

        The Train dataset

            2 subfolders:
                - set1
                - set2
            1 file:
                - obj.names: list of all possible classes

            Each set corresponds to a single complete Robotic Assisted Radical Prostatectomy (RARP) procedure.
            Each set contains frames filmed at 1FPS. Each frame has its corresponding .txt file with 
            the annotations. A frame can have more than one bounding box in it. However, some frames have
            none. In this case the .txt file will be empty.

        The Validation dataset 
        
            1 subfolder:
                - obj: frames + annotations
            1 file:
                - obj.names: list of all possible classes

            The structure of "obj" subfolder is similar to the structure of either set1 or set2. However, it 
            doesn't contain the full procedure (same frames are missing).

            WARNING:
                Not every frame (".jpg" image) has its corresponding annotation (".txt" file). And not every 
                annotation (".txt" file) has it's corresponding frame (".jpg" image).

        The Test dataset 
        
            2 parts:
                - test: contains frames (.jpg images)
                - test_labels: contains annotations (.txt files)

            It contains most of one full procedure (some frames are missing).
    """

    dataset = "ESAD"
    dataset_path = f"{args.download_output_dir}/{dataset}"

    gdrive_download_ids = {
        "train"         : "1CnYAzZRVEDGK1TycGBb8SnMgyvzeZrie",
        "val"           : "17rWwuWKFZFxQ0DTRs5cmUzU2Vb5PScol",
        "test"          : "1gho-oGzUbNgnZmBZ2GDKWWOcs1VI-z0O",
        "test_labels"   : "16srrq1NIso1mI2YKtHMIPyn5bZbcCyo3",
    }

    print("".join(["*"] * 50 ))
    print(f"Downloading dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):
        os.mkdir(f"{dataset_path}")
    else:
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")
        return
    
    files_to_download = ["train", "val", "test", "test_labels"]

    for file in files_to_download:
        print(f"Downloading ... {file}.zip")
        print(f"    From: https://drive.google.com/u/0/uc?id={gdrive_download_ids[file]}")
        print(f"    To: {dataset_path}")

        result = gdown.download(id=gdrive_download_ids[file], output=f"{dataset_path}/{file}.zip", quiet=True)
        assert result == f"{dataset_path}/{file}.zip", f"In download_{dataset}, error happened during: download {file}.zip"

        print(f"Unzipping ... {file}.zip")
        print(f"    To: {dataset_path}")

        result = subprocess.run(["unzip", "-uqq", f"{dataset_path}/{file}.zip", "-d", f"{dataset_path}"])
        assert result.returncode == 0, f"In download_{dataset}, error happened during: unzip {file}.zip"

        print(f"Removing ... {file}.zip")

        # remove zip files
        result = subprocess.run(["rm", f"{dataset_path}/{file}.zip"])
        assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}.zip"

        if file != "test_labels":
            print("")

    return


def prepare_ESAD(args):

    dataset = "ESAD"

    dataset_path = f"{args.download_output_dir}/{dataset}"
    output_path = f"{args.prepare_output_dir}/{dataset}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0

    # remove unnecessary files
    if (not args.preserve_original_structure):
        # remove obj.names files
        for folder in ["train", "val"]:
            result = subprocess.run(["rm", f"{dataset_path}/{folder}/obj.names"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}/obj.names"
    
    # process train/set1 train/set2 val/obj and test
    folders = [f"{dataset_path}/train/set1", f"{dataset_path}/train/set2", f"{dataset_path}/val/obj", f"{dataset_path}/test"]

    print("Creating images ...")

    image_counter = 0

    for i, folder in enumerate(folders):
        print(f"    Processing frames in ... {folder}")

        # Each folder contains images saved as (.jpg) and optionally corresponding annotation files (.txt).
        # If there is no annotation for a frame, the (.txt) will be empty. In order to avoid pre-operative
        # and post-operative frames we will take all images between the first and last non-empty 
        # annotation file, i.e. its (.txt) file has size > 0. 
        # In the test folder, all annoations are non-empty -> all images will be collected.

        images = glob.glob(f"{folder}/*.jpg")
        annotations = glob.glob(f"{folder}/*.txt" if folder.split("/")[-1] != "test" else f"{dataset_path}/test_labels/*.txt")

        first = 1000000
        last  = 0

        count_for_last_folder = image_counter

        # collect the frame number of first and last non-empty annotation
        for annotation in annotations:
            if (os.stat(annotation).st_size != 0):
                frame_number = int(annotation.split("/")[-1].split("_")[-1].split(".")[0])

                if (first > frame_number):
                    first = frame_number
                if (last < frame_number):
                    last = frame_number

        print(f"        First frame to take ... {first:5d}")
        print(f"        Last frame to take  ... {last:5d}")

        # save frames in-between first and last
        for img in images:
            frame_number = int(img.split("/")[-1].split("_")[-1].split(".")[0])

            if (first <= frame_number and frame_number <= last):
                image_counter += 1

                # relative path from top folder to the image will be the resulting name
                out_file_name = img.replace(f"{dataset_path}/", "").replace("/", "__")

                source = img
                destination = f"{output_path}/{out_file_name}"

                result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                        subprocess.run(["mv", f"{source}", f"{destination}"]) 
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {img}" if args.preserve_original_structure else \
                                            f"In prepare_{dataset}, error happened during: move {img}"

        print(f"        -----------------------------")
        print(f"        Total frames: {image_counter - count_for_last_folder:15d}")

        if i < 3:
            print("")

        if (not args.preserve_original_structure):
            # after we have processed all the frames remove the folder
            result = subprocess.run(["rm", "-r", f"{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"

    if (not args.preserve_original_structure):
        # remove train/val/test_labels folders, test get's removed beforehand because it has no subfolders
        for folder in ["train", "val", "test_labels"]:
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"

    # return the number of created images
    return image_counter


def download_LapGyn4(args):
    """
    STRUCTURE:

        # From readme.txt
        Naming Convention
        -----------------
            All datasets except Instrument_Count follow below naming convention:

            c_case#_v_video#_f_frame#.jpg

            case: a patient treatment case, i.e. an entire operation (may contain several videos)
            video: a video clip belonging to a certain case
            frame: a frame of a video

        Note: In the tree, all files are sorted by name, not by number.

        LapGyn4 (v1.2)
        ├── readme.txt
        |
        ├── Actions_on_Anatomy
        |   ├── Suturing_Other
        |   ├── Suturing_Ovary
        |   ├── Suturing_Uterus
        |   └── Suturing_Vagina
        |       ├── c_4_v_55_f_2203.jpg
        |       ...
        |       └── c_410_v_12686_f_1091.jpg
        |
        ├── Anatomical_Structures
        |   ├── Colon
        |   ├── Liver
        |   ├── Ovary
        |   ├── Oviduct
        |   └── Uterus
        |       ├── c_1_v_1132_f_2.jpg
        |       ...
        |       └── c_100_v_10643_f_1302.jpg
        |
        ├── Instrument_Count
        |   ├── 0
        |   ├── 1
        |   ├── 2
        |   └── 3
        |       ├── 31.jpg
        |       ...
        |       └── 21587.jpg
        |
        └── Surgical_Actions
            ├── Coagulation
            ├── Cutting_Cold
            ├── Cutting_HF
            ├── Dissection_Blunt
            ├── Injection
            ├── Sling_Hysterectomy
            ├── Suction_Irrigation
            └── Suturing
                ├── c_1_v_113587_f_13.jpg
                ...
                └── c_286_v_70367_f_5502.jpg

                
    Download links:
        - LapGyn4_v1.2
            "http://ftp.itec.aau.at/datasets/LapGyn4/downloads/v1_2/LapGyn4_v1.2.zip"


    DESCRIPTION:

        # From readme.txt
        The dataset contains 4 subfolders each representing an individual dataset:
            Dataset (#images):
            1) Actions_on_Anatomy        (4787)
                    Suturing_Other       (2110)
                    Suturing_Ovary        (715)
                    Suturing_Uterus       (940)
                    Suturing_Vagina      (1022)
            2) Anatomical_Structures     (2728)
                    Colon                 (295)
                    Liver                 (138)
                    Ovary                (1162)
                    Oviduct               (195)
                    Uterus                (938)
            3) Instrument_Count(*)      (21424)
                    0                    (5104)
                    1                    (5124)
                    2                    (5734)
                    3                    (5462)
            4) Surgical_Actions         (30682)
                    Coagulation          (3480)
                    Cutting_Cold         (1185)
                    Cutting_HF           (3752)
                    Dissection_Blunt     (1444)
                    Injection            (2119)
                    Sling_Hysterectomy   (2752)
                    Suction_Irrigation   (3036)
                    Suturing            (12914)

        No Train/Val/Test split.

        Frames are saved at 1 FPS for all cases.
        In the Instrument_Count folder images are randomized.

        Since some "Instrument Count" images come from Cholec80 we exclude the subfolder. We take all other images.
    """

    dataset = "LapGyn4"
    dataset_folder = f"{dataset}_v1.2"
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):	
        os.mkdir(f"{dataset_path}")	
    else:	
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")	
        return
    
    file = "LapGyn4_v1.2"
    link = "http://ftp.itec.aau.at/datasets/LapGyn4/downloads/v1_2/LapGyn4_v1.2.zip"

    print(f"Downloading ... {file}.zip")	
    print(f"    From: {link}")	
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-nc", "-q", "-O", f"{dataset_path}/{file}.zip", f"{link}"])
    assert result.returncode == 0, f"In download_{dataset}, error happened during: download {file}.zip"	

    print(f"Unzipping ... {file}.zip")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["unzip", "-uqq", f"{dataset_path}/{file}.zip", "-d", f"{args.download_output_dir}"])	
    assert result.returncode == 0, f"In download_{dataset}, error happened during: unzip {file}.zip"	

    print(f"Removing ... {file}.zip")	

    # remove zip files	
    result = subprocess.run(["rm", f"{dataset_path}/{file}.zip"])	
    assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}.zip"

    return


def prepare_LapGyn4(args):
    dataset = "LapGyn4"
    dataset_folder = f"{dataset}_v1.2"

    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
    output_path = f"{args.prepare_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0
    
    # remove unnecessary files/folders
    if (not args.preserve_original_structure):
        # remove "Readme.txt"
        for file in ["Readme.txt"]:
            result = subprocess.run(["rm", f"{dataset_path}/{file}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {file}"
        # remove Instrument_Count
        for folder in ["Instrument_Count"]:
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"

    # LapGyn4 consists of 4 individual smaller datasets, but we don't take Instrument_Count
    subdatasets = ["Actions_on_Anatomy", "Anatomical_Structures", "Surgical_Actions"]

    print("Creating images ...")

    image_counter = 0

    for i, subdataset in enumerate(subdatasets):
        if i > 0:
            print("")

        print(f"    Processing subdataset ... {subdataset}")

        count_for_last_subdataset = image_counter

        # each subdataset has subfolders which contain images saved as (.jpg)
        subfolders = glob.glob(f"{dataset_path}/{subdataset}/*")

        for subfolder in subfolders:
            print(f"        Processing videos in ... {subfolder}")

            images = glob.glob(f"{subfolder}/*.jpg")

            # save frames
            for img in images:
                image_counter += 1

                # relative path from top folder to the img will be the resulting name
                out_file_name = img.replace(f"{dataset_path}/", "").replace("/", "__")

                source = img
                destination = f"{output_path}/{out_file_name}"

                result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                         subprocess.run(["mv", f"{source}", f"{destination}"]) 
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {img}" if args.preserve_original_structure else \
                                               f"In prepare_{dataset}, error happened during: move {img}"

            if (not args.preserve_original_structure):
                # after we have processed all the frames remove the folder
                result = subprocess.run(["rm", "-r", f"{subfolder}"])
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {subfolder}"

        print(f"        -----------------------------")
        print(f"        Total frames: {image_counter - count_for_last_subdataset:15d}")
        
        if (not args.preserve_original_structure):
            # after we have processed all subfolders in the subdataset remove the subdataset
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{subdataset}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {subdataset}"        

    # return the number of created images
    return image_counter


def download_SurgicalActions160(args):
    """
    STRUCTURE:

        Naming Convention
        -----------------
            All videos have their names in the following format:
            
            -> Read # as "number"
            c_class#_v_video#.mp4

            class: one of the surgical action classes (e.g. Cutting, injection, ...)
            video: a video clip representing the corresponding class

        SurgicalActions160
        ├── README.txt
        ├── 01_abdominal_access
        ├── 02_injection
        ├── 03_cutting
        ├── 04_blunt_dissection
        ├── 05_dissection_thermal
        ├── 06_irrigation
        ├── 07_coagulation
        ├── 08_suction
        ├── 09_needle_positioning
        ├── 10_needle_puncture
        ├── 11_knot_pushing
        ├── 12_knotting
        ├── 13_thread_cut
        ├── 14_sling-in
        ├── 15_endobag-in
        └── 16_endobag-out
            ├── 16_01.mp4
            ...  
            └── 16_10.mp4

            
    Download links:
        - SurgicalActions160
            "http://ftp.itec.aau.at/datasets/SurgicalActions160/downloads/SurgicalActions160.zip"

                
    DESCRIPTION:

        This dataset consists of 160 short videos (in .mp4 format) showing typical surgical actions in 
        gynecologic laparoscopy. They are divided into subfolders depending on the action they represent. In
        total there are 16 surgical action classes and therefore 16 subfolders, each containing 10 videos.

        # From README.txt
        The classes are as follows:
            01: Abdominal access      Initial access to the abdomen (puncture)
            02: Injection             Injection of anesthetization liquid
            03: Cutting               Cutting tissue with scissors
            04: Blunt dissection      Dissection of tissue with blunt instruments
            05: Dissection (thermal)  Thermal dissection of tissue with an electrosurgical instrument
            06: Irrigation            Cleaning of the operation area with the Suction and Irrigation Tube instrument
            07: Coagulation           Coagulation of tissue with the Coagulation Forceps instrument
            08: Suction               Cleaning of the operation area with the Suction and Irrigation Tube instrument
            09: Needle positioning    Bringing the needle into right position and orientation
            10: Needle puncture       Puncturing with the suturing needle
            11: Knot pushing          Pushing an externally tied knot to the suturing area with the Knot Pusher instrument
            12: Knotting              Tying a knot during suturing (inside of the patient)
            13: Thread cut            Cutting a thread after suturing
            14: Sling-In              Insertion of the Dissection Sling instrument
            15: Endobag-In            Insertion of the Endobag tool
            16: Endobag-Out           Removal of the Endobag tool

        Video clips were extracted from 59 different recordings and have a resolution of 427×240 pixels.
        Their duration is approximately 5s each. In total, the dataset consists of 19181 frames.

        Pre-processing: All subfolders will be deleted and videos downsampled to 1 FPS.

        Note: Videos (.mp4 files) will be converted to a sequence of images at the rate of 1FPS. The naming
              convention we will use is the following:

              {class_name}__{video_name}__frame#{frame_number},

              where:
                class_name - one of the surgical actions classes
                video_name - name of the video containing the frame
                frame_number - frame number in order of extraction (i.e. if we extract frames at 1FPS,
                               frame 0 will have number 1, frame 25 number 2, frame 50 number 3 and so on)
    """

    dataset = "SurgicalActions160"
    dataset_folder = dataset
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
	
    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):		
        os.mkdir(f"{dataset_path}")		
    else:		
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")		
        return	
    	
    file = "SurgicalActions160"	
    link = "http://ftp.itec.aau.at/datasets/SurgicalActions160/downloads/SurgicalActions160.zip"

    print(f"Downloading ... {file}.zip")		
    print(f"    From: {link}")		
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-nc", "-q", "-O", f"{dataset_path}/{file}.zip", f"{link}"])
    assert result.returncode == 0, f"In download_{dataset}, error happened during: download {file}.zip"

    print(f"Unzipping ... {file}.zip")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["unzip", "-uqq", f"{dataset_path}/{file}.zip", "-d", f"{dataset_path}"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: unzip {file}.zip"

    print(f"Removing ... {file}.zip")		
    # remove zip files		
    result = subprocess.run(["rm", f"{dataset_path}/{file}.zip"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}.zip"

    return


def prepare_SurgicalActions160(args):
    dataset = "SurgicalActions160"
    dataset_folder = dataset

    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
    output_path = f"{args.prepare_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0
    
    # remove unnecessary files/folders
    if (not args.preserve_original_structure):
        # remove "README.txt"
        for file in ["README.txt"]:
            result = subprocess.run(["rm", f"{dataset_path}/{file}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {file}"

    # SurgicalActions160 has 16 subfolders each representing 1 surgical action class 
    subfolders = glob.glob(f"{dataset_path}/[[:digit:]][[:digit:]]_*") # everything excpet the README.txt

    print("Creating images ...")

    image_counter = 0

    for i, subfolder in enumerate(subfolders):

        print(f"    Processing videos in ... {subfolder}")

        # each subfolder has 10 short videos (.mp4 files)
        videos = glob.glob(f"{subfolder}/*.mp4")

        for video_path in videos:
            out_file_name = (video_path.replace(f"{dataset_path}/", "") + "/frame").replace("/", "__")
            destination = f"{output_path}/{out_file_name}"

            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", f"{video_path}", "-vf", "fps=1", f"{destination}#%01d.jpg"])


        if (not args.preserve_original_structure):
            # after we have processed all the frames remove the folder
            result = subprocess.run(["rm", "-r", f"{subfolder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {subfolder}"

    image_counter = len(os.listdir(output_path))

    print(f"    -----------------------------")
    print(f"    Total frames: {image_counter:15d}")
            
    # return the number of created images
    return image_counter


def download_GLENDA(args):
    """
    STRUCTURE:
        Naming Convention
        -----------------
            # From README.txt

            Placeholders
            +-------------+------------------------------+
            | Placeholder | Description                  |
            +-------------+------------------------------+
            | v_vID       | Video (v) with ID (vID)      |
            +-------------+------------------------------+
            | f_fID       | Frame (f) with ID (fID)      |
            +-------------+------------------------------+
            | a_aID       | Annotation (a) with ID (aID) |
            +-------------+------------------------------+
            | s_FROM-TO   | Video sequence (s) with      |
            |             | starting frame number (FROM) |
            |             | and end frame number (TO)    |
            +-------------+------------------------------+


            Folders
            +----------------------------------------------------------+--------------------------------------------------------------------------------------------+
            | Folder                                                   | Description                                                                                |
            +----------------------------------------------------------+--------------------------------------------------------------------------------------------+
            | DS/pathology/annotations                                 | Binary annotation images (1 per annotation) of structure:                                  |
            |                                                          | v_vID_s_FROM-TO/f_fID/CLASS_a_aID.png                                                      |
            +----------------------------------------------------------+--------------------------------------------------------------------------------------------+
            | DS/*pathology/images                                     | Video frames of structure:                                                                 |
            |                                                          | v_vID_s_FROM-TO/f_fID_a.jpg ("_a" indicates an existing annotation for that frame)         |
            +----------------------------------------------------------+--------------------------------------------------------------------------------------------+
            | statistics/*                                             | Image and Annotation statistics:                                                           |
            |                                                          | annotations/images per class, max. annotations per image, ..                               |
            +----------------------------------------------------------+--------------------------------------------------------------------------------------------+


        GLENDA_v1.0
        ├── Readme.txt
        |
        ├── statistics
        |   ├── annotations.csv
        |   └── images.csv
        |
        └── DS
            ├── no_pathology
            │   └── frames
            |       ├── v_2506_s_0-95
            |       |   ├── f_0.jpg
            |       |   ...
            |       |   └── f_94.jpg
            |       ...
            |       └── v_5675_s_547-736
            |           ├── f_547.jpg
            |           ...
            |           └── f_735.jpg          
            |
            └── pathology
                ├── annotations
                |    ├── v_7_s_1299-1299
                |    |    └── f_1299
                |    |        └── ovary_a_848.png
                |    ...
                |    └── v_9755_s_3548-3624
                |        └── f_3595                         #not every frame is annotated
                |            ├── peritoneum_a_837.png
                |            ...
                |            └── peritoneum_a_841.png
                |
                └── frames
                    ├── v_7_s_1299-1299
                    |    └── f_1299.jpg
                    ...
                    └── v_9755_s_3548-3624
                        ├── f_3548.jpg
                        ...
                        └── f_3623.jpg 

                        
    Download links:
        - GLENDA v1.0
            "http://ftp.itec.aau.at/datasets/GLENDA/v1_0/downloads/GLENDA_v1.0.zip"


    DESCRIPTION:

        Currently version 1.5 and 1.0 are available. Since we only care about unannotated images,
        we take v1.0.

        The dataset has 2 main subfolders:
            1) pathology
            2) no_pathology

        Each of them has multiple other subfolders containing frames from a video sequence. Subolders in 1) 
        mostly contain 1 frame only, but not all of them.

        There are in total 5 classes:
            -> four based on position of pathology (peritoneum, ovary, uterus, DIE - deep infiltrating
               endometriosis) 
            -> and no pathology

        The sequences will be downsampled by taking every 25th frame.
    """

    dataset = "GLENDA"
    dataset_folder = f"{dataset}_v1.0"
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
	
    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):		
        os.mkdir(f"{dataset_path}")		
    else:		
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")		
        return	
    	
    file = f"{dataset_folder}"	
    link = "http://ftp.itec.aau.at/datasets/GLENDA/v1_0/downloads/GLENDA_v1.0.zip"

    print(f"Downloading ... {file}.zip")		
    print(f"    From: {link}")		
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-nc", "-q", "-O", f"{dataset_path}/{file}.zip", f"{link}"])
    assert result.returncode == 0, f"In download_{dataset}, error happened during: download {file}.zip"

    print(f"Unzipping ... {file}.zip")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["unzip", "-uqq", f"{dataset_path}/{file}.zip", "-d", f"{args.download_output_dir}"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: unzip {file}.zip"

    print(f"Removing ... {file}.zip")		
    # remove zip files		
    result = subprocess.run(["rm", f"{dataset_path}/{file}.zip"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}.zip"

    return


def prepare_GLENDA(args):
    dataset = "GLENDA"
    dataset_folder = f"{dataset}_v1.0"

    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
    output_path = f"{args.prepare_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0
    
    # remove unnecessary files/folders
    if (not args.preserve_original_structure):
        # remove "Readme.txt"
        for file in ["Readme.txt"]:
            result = subprocess.run(["rm", f"{dataset_path}/{file}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {file}"

        # remove "statistics" and "DS/pathology/annotations"
        for folder in ["statistics", "DS/pathology/annotations"]:
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"


    # DS folder has 2 main subfolders: pathology/no_pathology
    subfolders = glob.glob(f"{dataset_path}/DS/*")

    print("Creating images ...")

    image_counter = 0

    for i, subfolder in enumerate(subfolders):
        print(f"    Processing videos in ... {subfolder}")

        # DS/pathology/frames/* | DS/no_pathology/frames/*
        video_sequence_folders = glob.glob(f"{subfolder}/frames/*")

        for vsf in video_sequence_folders:
            # get the frames
            frames = glob.glob(f"{vsf}/*")

            start_frame = 0
            frames_to_take = frames[start_frame:-1:25]

            # save frames
            for frame in frames_to_take:
                image_counter += 1

                # relative path from top folder to the img will be the resulting name
                out_file_name = frame.replace(f"{dataset_path}/", "").replace("/", "__")

                source = frame
                destination = f"{output_path}/{out_file_name}"

                result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                         subprocess.run(["mv", f"{source}", f"{destination}"]) 
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {frame}" if args.preserve_original_structure else \
                                               f"In prepare_{dataset}, error happened during: move {frame}"

            if (not args.preserve_original_structure):
                # after we have processed all the frames remove the vsf
                result = subprocess.run(["rm", "-r", f"{vsf}"])
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {vsf}"

        if (not args.preserve_original_structure):
            # after we have processed all video sequences remove the pathology / no_pathology subfolder
            result = subprocess.run(["rm", "-r", f"{subfolder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {subfolder}"

        # after processing everything remove DS folder
    if (not args.preserve_original_structure):
        result = subprocess.run(["rm", "-r", f"{dataset_path}/DS"])
        assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {dataset_path}/DS"

    print(f"    -----------------------------")
    print(f"    Total frames: {image_counter:15d}")
            
    # return the number of created images
    return image_counter


def download_hSDB_instrument(args):
    """
    STRUCTURE:
        # Note we only download gastrectomy videos, not cholecystectomy ones.

        hSDB-instrument
        ├── gastric_real_train.json
        ├── gastric_real_val.json
        ├── gastric_real_train
        └── gastric_real_val
            ├── 0000000000.jpg
            ...  
            └── 0000005931.jpg

    DESCRIPTION:

        There are 4 main files/subfolders:
            1) gastric_real_train.json
                - contains annotations for the train dataset
            2) gastric_real_val.json
                - contains annotations for the validation dataset
            3) gastric_real_train
                - contains raw train images (in .jpg format)
            4) gastric_real_val
                - contains raw validation images (in .jpg format)

        This dataset consists of laparoscopic cholecystectomy vidoes and gastrectomy videos,
        which are downloaded separately. Since laparoscopic videos come from Cholec80
        we only process the gastrectomy ones. The data has already been pre-processed to 1FPS.

        Moreover, the dataset has real and synthetic data. However, we are only interested in
        real data.
        
        Note: We could not find the test dataset. We download "gastric_real.tar" file, which
              only contains train and val data.

    """

    dataset = "hSDB-instrument"
    dataset_folder = dataset
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
	
    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):		
        os.mkdir(f"{dataset_path}")		
    else:		
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")		
        return	
    
    file = "gastric_real.tar"
    #file = "gastric_real_syn_dr.tar"
    link = r'https://drive.google.com/u/0/uc?id=10nV_OolHANtxLSzTyw8GnLv2flXd6jVw&export=download&confirm=t&uuid=fbe93740-042f-47ab-9c17-eabe700d65ad'
    #link = r'https://drive.google.com/u/0/uc?id=1aGiuPhuWJcBD2DACirV2CkMg9nl92QdN&export=download&confirm=t&uuid=fbe93740-042f-47ab-9c17-eabe700d65ad'

    print(f"Downloading ... {file}")
    print(f"    From: {link}")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-q", "-O", f"{dataset_path}/{file}", f"{link}"])
    assert result == 0, f"In download_{dataset}, error happened during: download {file}"

    print(f"Extracting ... {file}")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["tar", "xf", f"{dataset_path}/{file}", "-C", f"{dataset_path}", "--strip-components", "1"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: tar {file}"

    print(f"Removing ... {file}")		
    # remove tar file
    result = subprocess.run(["rm", f"{dataset_path}/{file}"])		
    assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}"

    return


def prepare_hSDB_instrument(args):
    dataset = "hSDB-instrument"

    dataset_path = f"{args.download_output_dir}/{dataset}"
    output_path = f"{args.prepare_output_dir}/{dataset}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0

    # remove unnecessary files
    if (not args.preserve_original_structure):
        # remove .json files
        for file in ["gastric_real_train.json", "gastric_real_val.json"]:
            result = subprocess.run(["rm", f"{dataset_path}/{file}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {file}"
    
    # process gastric_real_train and gastric_real_val
    folders = [f"{dataset_path}/gastric_real_train", f"{dataset_path}/gastric_real_val"]

    print("Creating images ...")

    image_counter = 0

    for i, folder in enumerate(folders):
        print(f"    Processing frames in ... {folder}")

        # Each folder contains images saved as (.jpg). Since the images are already subsampled to 1FPS we take all of them.
        images = glob.glob(f"{folder}/*")

        for img in images:
            image_counter += 1

            # relative path from top folder to the image will be the resulting name
            out_file_name = img.replace(f"{dataset_path}/", "").replace("/", "__")

            source = img
            destination = f"{output_path}/{out_file_name}"

            result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                     subprocess.run(["mv", f"{source}", f"{destination}"]) 
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {img}" if args.preserve_original_structure else \
                                           f"In prepare_{dataset}, error happened during: move {img}"

        if (not args.preserve_original_structure):
            # after we have processed all the frames remove the folder
            result = subprocess.run(["rm", "-r", f"{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"

    print(f"    -----------------------------")
    print(f"    Total frames: {image_counter:15d}")

    # return the number of created images
    return image_counter


def download_data(email, password, local_folder):
    """
    Helper function for download_HeiCo function.
    """

    # login to Synapse
    syn = synapseclient.login(email=email, password=password, rememberMe=True)

    # download all the files in folder files_synapse_id to a local folder
    project_id = "syn21903917" # this is the project id of the files.
    all_files = synapseutils.syncFromSynapse(syn, entity=project_id, path=local_folder)


def download_HeiCo(args):
    """
    IMPORTANT:
        To download the dataset a "synapse account" is required: https://www.synapse.org/

        We follow the instructions listed at: https://www.synapse.org/#!Synapse:syn21903917/wiki/601992

        And using the provided script download the **COMPLETE dataset**, not the ROBUST-MIS 2019 challenge version.

    STRUCTURE:

        HeiCo
        ├── Proctocolectomy
        |   ├── 1
        |   |   ├── Proctocolectomy_1_Device.csv
        |   |   ├── Proctocolectomy_1_Phase.csv
        |   |   ├── Proctocolectomy_1.avi
        |   |   └── Instrument segmentations
        |   |       ├── ...
        |   |       ├── i
        |   |       |   ├── raw.png
        |   |       |   ├── 10s_video.zip
        |   |       |   └── instrument_instances.png (*only if the instrument is visible)
        |   |       ... 
        |   |       └── ...
        |   ...
        |   └── 10
        |        ├── ...
        |        ...
        |
        ├── Rectal Resection
        |   ├── 1
        |   ...
        |   └── 10
        |
        └── Sigmoid Resection
            ├── 1
            ...
            └── 10

    DESCRIPTION:
        # taken from the HeiCo paper itself
        The data is organized in a 4-level folder structure. The first level represents the surgery type (proctocolectomy,
        rectal resection and sigmoid resection). In the next lower level, the folder names are integers ranging from 1-10
        and represent procedure numbers. Each folder in this second level (corresponding to a surgery type p and procedure
        number i) contains the raw data (laparoscopic video as .avi file and device data as .csv file), the surgical phase
        annotations (as .csv file) and a set of subfolders numbered from 1 to Np,i where Np,i is the number of the frames
        for which instrument segmentations were acquired. The final 4th level represents individual video frames and 
        contains the video frame itself (raw.png) and a 10 second video (10s_video.zip) of the 250 preceding frames in RGB
        format. If instruments are visible in an image frame, the folder contains an additional file called 
        “instrument_instances.png”, which represents the instrument segmentations.

        Pre-processing:
        In each of the first level folders (which respresent different surgery types), we process each of the 10 
        subfolders (containing individual operation files) and extract frames from the full ".avi" videos at 1FPS.
        Everything else is ignored.

        **NOTE** This dataset is very large. (~600 GB)
    """

    dataset = "HeiCo"
    dataset_folder = dataset
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
	
    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (args.synapse_email == ''):
        print(f"WARNING: Synapse email is NOT set. Download of {dataset} dataset will be skipped.")
        return
    if (args.synapse_password == ''):
        print(f"WARNING: Synapse password is NOT set. Download of {dataset} dataset will be skipped.")
        return

    if (not os.path.exists(f"{dataset_path}")):		
        os.mkdir(f"{dataset_path}")		
    else:		
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")		
        return
    
    download_data(args.synapse_email, args.synapse_password, dataset_path)

    return


def prepare_HeiCo(args):
    dataset = "HeiCo"
    dataset_folder = dataset

    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
    output_path = f"{args.prepare_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0

    # HeiCo has 3 main subfolders each representing one type of surgery:
    # Proctocolectomy/Rectal Resection/Sigmoid Resection
    surgery_types = os.listdir(dataset_path)
    surgery_types.remove("SYNAPSE_METADATA_MANIFEST.tsv")

    print("Creating images ...")

    image_counter = 0

    for i, surgery_type in enumerate(surgery_types):
        print(f"    Processing surgery type ... {surgery_type}")

        count_for_last_surgery_type = image_counter

        # each surgery_type contains 10 subfolders denoting different procedures
        procedures = glob.glob(f"{dataset_path}/{surgery_type}/*")
        procedures.remove(f"{dataset_path}/{surgery_type}/SYNAPSE_METADATA_MANIFEST.tsv")

        for j, procedure in enumerate(procedures):
            print(f"        Processing procedure ... {str(j):2s}")

            raw_video_file = glob.glob(f"{procedure}/*.avi")[0]

            # each of these subfolders contains multiple files including .avi video file of the whole procedure
            out_file_name = (raw_video_file.replace(f"{dataset_path}/", "") + "/frame").replace("/", "__")
            destination = f"{output_path}/{out_file_name}"

            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", f"{raw_video_file}", "-vf", "fps=1", f"{destination}#%05d.jpg"])

            if (not args.preserve_original_structure):
                # after we have processed all the frames remove the procedure folder
                result = subprocess.run(["rm", "-r", f"{procedure}"])
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {procedure}"
    
        if (not args.preserve_original_structure):
            # after we have processed all video sequences remove the surgery type folder
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{surgery_type}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {dataset_path}/{surgery_type}"

        image_counter = len(os.listdir(output_path))
        image_counter -= (i+1) if not args.preserve_original_structure else 0

        print(f"        ---------------------------")
        print(f"        Total frames: {image_counter-count_for_last_surgery_type:13d}")

    # remove unnecessary files
    result = subprocess.run(["rm", f"{dataset_path}/SYNAPSE_METADATA_MANIFEST.tsv"])
    assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {dataset_path}/SYNAPSE_METADATA_MANIFEST.tsv"
            
    # return the number of created images
    return image_counter


def download_PSI_AVA(args):
    """
    STRUCTURE:

        PSI-AVA
        ├── TAPIR_trained_models
        ├── def_DETR_box_ftrs
        |
        ├── images_8_frames_per_second
        |   ├── keyframes
        |   ├── RobotSegSantaFe_v3_dense.json
        |   ├── RobotSegSantaFe_v3_dense_fold1.json
        |   └── RobotSegSantaFe_v3_dense_fold2.json 
        |
        └── keyframes
            ├── CASE001
            ├── CASE002
            ...
            └── CASE021
                ├── 00000.jpg
                ├── 00001.jpg
                ...
                └── 11870.jpg

                
    Download links:
        - PSI-AVA
            "http://157.253.243.19/PSI-AVA/"
        - PSI-AVA keyframes 1FPS only
            "http://157.253.243.19/PSI-AVA/keyframes"


    DESCRIPTION:
        1) TAPIR_trained_models
            -> contains TAPIR pre-trained checkpoints for each task separately

        2) def_DETR_box_ftrs
            -> DETR bounding box predictions for instruments

        3) images_8_frames_per_second
            a) keyframes
                -> Contains full surgery videos subsampled to 8 FPS (We believe that originally surgeries were filmed at 45 FPS.)
                -> Same structure as 4)
            b) RobotSegSantaFe_v3_dense.json
                -> Annotations for the full dataset.
            c) RobotSegSantaFe_v3_dense_fold1.json
                -> Annotations for fold1 of the dataset.
            d) RobotSegSantaFe_v3_dense_fold1.json
                -> Annotations for fold2 of the dataset.

        4) keyframes
            -> Contains full surgery videos subsampled to 1 FPS.

        For more info about the dataset please visit "https://github.com/BCV-Uniandes/TAPIR".

        We don't separate between train and validation sets and instead simply take all frames in "4) keyframes".
    """

    dataset = "PSI_AVA"
    dataset_folder = dataset
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"
	
    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):		
        os.mkdir(f"{dataset_path}")		
    else:		
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")		
        return	
    
    # PSI-AVA link
    link = "http://157.253.243.19/PSI-AVA/" if args.PSI_AVA_download_all else "http://157.253.243.19/PSI-AVA/keyframes"

    print(f"Downloading ...")		
    print(f"    From: {link}")		
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-r", "-q", "--cut-dirs", "1", "--no-parent", "-nH", "-P", f"{dataset_path}", f"{link}"])
    assert result.returncode == 0, f"In download_{dataset}, error happened during: download {dataset}"

    return


def prepare_PSI_AVA(args):
    dataset = "PSI_AVA"

    dataset_path = f"{args.download_output_dir}/{dataset}"
    output_path = f"{args.prepare_output_dir}/{dataset}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0

    # remove unnecessary files/folders: all subfolders except keyframes
    if (not args.preserve_original_structure):
        # remove unnecessary files: index.html
        files_to_remove = [name for name in os.listdir(dataset_path) if os.path.isfile(f"{dataset_path}/{name}")]

        for file in files_to_remove:
            result = subprocess.run(["rm", f"{dataset_path}/{file}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {file}"

        # all subfolders except keyframes
        folders_to_remove = [name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}") and name != "keyframes"]

        for folder in folders_to_remove:
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"
    
    # 8 surgeries in total, naming convencion: CASE{surgery_number}, e.g. CASE001
    surgeries = glob.glob(f"{dataset_path}/keyframes/CASE*")

    print("Creating images ...")

    image_counter = 0

    for i, surgery in enumerate(surgeries):
        print(f"    Processing frames in ... {surgery}")

        # Each folder contains images saved as (.jpg). Since the images are already subsampled to 1FPS we take all of them.
        images = glob.glob(f"{surgery}/*.jpg")

        for img in images:
            image_counter += 1

            # relative path from top folder to the image will be the resulting name
            out_file_name = img.replace(f"{dataset_path}/", "").replace("/", "__")

            source = img
            destination = f"{output_path}/{out_file_name}"

            result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                     subprocess.run(["mv", f"{source}", f"{destination}"]) 
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {img}" if args.preserve_original_structure else \
                                           f"In prepare_{dataset}, error happened during: move {img}"

        if (not args.preserve_original_structure):
            # after we have processed all the frames remove the surgery folder
            result = subprocess.run(["rm", "-r", f"{surgery}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {surgery}"

    if (not args.preserve_original_structure):
        # after we have processed all surgeris remove the "keyframes" folder
        result = subprocess.run(["rm", "-r", f"{dataset_path}/keyframes"])
        assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove keyframes"


    print(f"    -----------------------------")
    print(f"    Total frames: {image_counter:15d}")

    # return the number of created images
    return image_counter


def download_DSAD(args):
    """
    STRUCTURE:

        DSAD
        ├── abdominal_wall
        ├── colon
        ├── inferior_mesenteric_artery
        ├── intestinal_veins
        ├── liver
        ├── multilabel
        ├── pancreas
        ├── small_intestine
        ├── spleen
        ├── stomach
        ├── ureter
        └── vesicular_glands
            ├── 03
            ├── 04
            ...
            └── 29
                ├── anno_1
                ├── anno_2
                ├── anno_3
                ├── merged
                ├── image00.png
                ...
                ├── image61.png
                ├── mask00.png
                ...
                ├── mask61.png
                └── weak_labels.csv


    Download links:
        - DSAD
            "https://springernature.figshare.com/ndownloader/files/38494425"

                
    DESCRIPTION:
        # Taken from the paper introducing the dataset.

        The data is organized in a 3-level folder structure. The first level is composed of twelve subfolders, 
        one for each organ/anatomical structure (abdominal_wall, colon, inferior_mesenteric_artery, intestinal_veins, 
        liver, pancreas, small_intestine, spleen, stomach, ureter and vesicular_glands) and one for the multi-organ dataset 
        (multilabel).

        Each folder contains 20 to 23 subfolders for the diferent surgeries that the images have been extracted from. 
        The subfolder nomenclature is derived from the individual index number of each surgery. 
        Each of these folders contains two versions of 5 to 91 PNG-files, one raw image that has been extracted from
        the surgery video file and one image that contains the mask of the expert-reviewed semantic segmentation 
        (black=background, white=segmentation). The raw images are named imagenumber.png, (e.g. image23.png),
        the masks are named masknumber.png (e.g. mask23.png). In the multilabel folder there are separate masks for
        each of the considered structures visible on the individual image (e.g. masknumber_stomach.png). The image
        indices always match for associated images. Each surgery- and organ-specific folder furthermore contains a CSV
        file named weak_labels.csv that contains all information about the visibility of the eleven regarded organs
        in the respective images. The columns in these CSV files are ordered alphabetically: Abdominal wall, colon,
        inferior mesenteric artery, intestinal veins, liver, pancreas, small intestine, spleen, stomach, ureter and
        vesicular glands.

        Additionally, the folders anno_1, anno_2, anno_3 and merged can be accessed from the surgery- and 
        organ-specifc subfolders. These folders contain the masks generated by the diferent annotators and the
        automatically generated merged version of the masks, each in PNG format.

        Note: anno_1, anno_2, anno_3 and merged are initial masks which were used to create masknumber.png files.
        
        Pre-processing:
        -> Since all images have already been downsampled, in each subfolder we take all the imagenumber.png files.
        -> Since multilabel subfolder contains the same images as stomach subfolder we skip it.
    """

    dataset = "DSAD"
    dataset_folder = dataset
    dataset_path = f"{args.download_output_dir}/{dataset_folder}"

    print("".join(["*"] * 50 ))	
    print(f"Downloading dataset ... {dataset}")	
    print("".join(["*"] * 50 ))

    if (not os.path.exists(f"{dataset_path}")):	
        os.mkdir(f"{dataset_path}")	
    else:	
        print(f"WARNING: Folder \"{dataset_path}\" already exists! Download of {dataset} dataset will be skipped.")	
        return
    
    file = "DSAD.zip"
    link = "https://springernature.figshare.com/ndownloader/files/38494425"

    print(f"Downloading ... {file}")	
    print(f"    From: {link}")	
    print(f"    To: {dataset_path}")

    result = subprocess.run(["wget", "-nc", "-q", "-O", f"{dataset_path}/{file}", f"{link}"])
    assert result.returncode == 0, f"In download_{dataset}, error happened during: download {file}"	

    print(f"Unzipping ... {file}")
    print(f"    To: {dataset_path}")

    result = subprocess.run(["unzip", "-uqq", f"{dataset_path}/{file}", "-d", f"{dataset_path}"])	
    assert result.returncode == 0, f"In download_{dataset}, error happened during: unzip {file}"	

    print(f"Removing ... {file}")	

    # remove zip files	
    result = subprocess.run(["rm", f"{dataset_path}/{file}"])	
    assert result.returncode == 0, f"In download_{dataset}, error happened during: remove {file}"

    return


def prepare_DSAD(args):
    dataset = "DSAD"

    dataset_path = f"{args.download_output_dir}/{dataset}"
    output_path = f"{args.prepare_output_dir}/{dataset}"

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    if (args.preserve_original_structure and not os.path.exists(f"{output_path}")):
        os.mkdir(f"{output_path}")
    elif (args.preserve_original_structure and os.path.exists(f"{output_path}")):
        print(f"WARNING: Folder \"{output_path}\" already exists! Preparation for {dataset} dataset will be skipped.")
        return 0

    # remove unnecessary folders: multilabel subfolder contains the same images as stomach
    if (not args.preserve_original_structure):
        folders_to_remove = ["multilabel"]

        for folder in folders_to_remove:
            result = subprocess.run(["rm", "-r", f"{dataset_path}/{folder}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {folder}"

    # DSAD consists of 11 subfolders each representing an anatomical structure
    anatomies = glob.glob(f"{dataset_path}/*")
    anatomies.remove(f"{dataset_path}/multilabel")

    print("Creating images ...")

    image_counter = 0

    for i, anatomy in enumerate(anatomies):

        print(f"    Processing images in ... {anatomy}")

        # each anatomy folder has subfolders, named after the surgery number the images inside come from
        surgeries = glob.glob(f"{anatomy}/*")

        for surgery in surgeries:
            # each surgery folder contains annotations and images saved as image{image_number}.png
            images = glob.glob(f"{surgery}/image*")

            # save frames
            for img in images:
                image_counter += 1

                # relative path from top folder to the img will be the resulting name
                out_file_name = img.replace(f"{dataset_path}/", "").replace("/", "__")

                source = img
                destination = f"{output_path}/{out_file_name}"

                result = subprocess.run(["cp", f"{source}", f"{destination}"]) if args.preserve_original_structure else \
                         subprocess.run(["mv", f"{source}", f"{destination}"]) 
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: copy {img}" if args.preserve_original_structure else \
                                               f"In prepare_{dataset}, error happened during: move {img}"

            if (not args.preserve_original_structure):
                # after we have processed all the frames remove the folder
                result = subprocess.run(["rm", "-r", f"{surgery}"])
                assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {surgery}"
        
        if (not args.preserve_original_structure):
            # after we have processed all subfolders in the subdataset remove the subdataset
            result = subprocess.run(["rm", "-r", f"{anatomy}"])
            assert result.returncode == 0, f"In prepare_{dataset}, error happened during: remove {anatomy}"


    print(f"    -----------------------------")
    print(f"    Total frames: {image_counter:15d}")    

    # return the number of created images
    return image_counter


def main(args):

    seed = 6304 # Do not change!

    # The datasets are ordered by their size: from highest to lowest.
    # When downloading all datasets and pre-processing in-place this
    # saves memory as all unnecessary data gets deleted before starting
    # the download of the next dataset.
    selected_datasets = {
        "HeiCo_dataset": args.HeiCo_dataset,
        "PSI_AVA": args.PSI_AVA,
        "ESAD": args.ESAD,
        "LapGyn4": args.LapGyn4,
        "hSDB_instrument": args.hSDB_instrument,
        "Dresden_dataset": args.DSAD,
        "GLENDA": args.GLENDA,
        "SurgicalActions160": args.SurgicalActions160,
    }

    download_functions = {
        "ESAD": download_ESAD,
        "LapGyn4": download_LapGyn4,
        "SurgicalActions160": download_SurgicalActions160,
        "GLENDA": download_GLENDA,
        "hSDB_instrument": download_hSDB_instrument,
        "HeiCo_dataset": download_HeiCo,
        "PSI_AVA": download_PSI_AVA,
        "Dresden_dataset": download_DSAD,
    }

    prepare_functions = {
        "ESAD": prepare_ESAD,
        "LapGyn4": prepare_LapGyn4,
        "SurgicalActions160": prepare_SurgicalActions160,
        "GLENDA": prepare_GLENDA,
        "hSDB_instrument": prepare_hSDB_instrument,
        "HeiCo_dataset": prepare_HeiCo,
        "PSI_AVA": prepare_PSI_AVA,
        "Dresden_dataset": prepare_DSAD,
    }

    image_counts = {}

    # download/prepare the datasets
    for dataset, is_selected in selected_datasets.items():
        if (args.all or is_selected):
            if(args.mode == "dnp"):
                download_functions[dataset](args)
                print("") # to introduce spacing in the output 
                random.seed(seed)
                image_counts[dataset] = prepare_functions[dataset](args)
            elif(args.mode == "d"):
                download_functions[dataset](args)
                image_counts[dataset] = 0
            else:
                random.seed(seed)
                image_counts[dataset] = prepare_functions[dataset](args)

            print("")

    # print #images for each dataset
    if ((args.mode == "dnp" or args.mode == "p") and len(image_counts.items()) > 1):
        total = 0

        print("*" * 50)
        print("*" * 50)
        print("")
        print("Summary:")

        for key, value in image_counts.items():
            if (value > 0):
                print("{:>25}: {:>10}".format(key, value))
            total += value

        print("-" * 50)
        print("{:>25}: {:>10}".format("Total", total))

    #TODO: Asserts on #images

    return


if __name__ == '__main__':
    args = get_argument_parser()
    args = args.parse_args()

    if args.download_output_dir:
        Path(args.download_output_dir).mkdir(parents=True, exist_ok=True)

    args.preserve_original_structure = args.download_output_dir != args.prepare_output_dir

    if args.prepare_output_dir and args.preserve_original_structure:
        Path(args.prepare_output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
