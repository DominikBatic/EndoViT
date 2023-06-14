import os
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

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
                    Path where the output file containing per class weights will be saved.
                    """)

FLAGS, unparsed = parser.parse_known_args()

#################################################################################################
######################################## Initialization #########################################
#################################################################################################

data_dir            = FLAGS.data_dir
output_dir          = FLAGS.output_dir

# output file path
output_file_path       = os.path.join(output_dir, 'per_class_weights.txt')
          
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

    total_frames = len(triplet_labels)
    counts = triplet_labels.sum(axis=0)
    assert counts.shape == (100,), f"ERROR: Expected count shape (100,) got {counts.shape} for video '{video}'."
    per_video_counts[videos[i]] = np.concatenate(
                                    (
                                        counts[np.newaxis, :], 
                                        (total_frames - counts)[np.newaxis, :],

                                    ), axis=0) # save number of positive and negative occurances of each class per video
    
    assert np.all(per_video_counts[videos[i]].sum(axis=0) - total_frames == 0), "Total counts don't match up!"

    print(f"Processed Video {i+1:2d}/{len(videos)}")
    print(f"\t -> Total frames: {total_frames}")

print("\nSaving results to {output_file_path} ...")

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

    train_count = np.zeros((2, 100), dtype=int)
    val_count = np.zeros((2, 100), dtype=int)
    test_count = np.zeros((2, 100), dtype=int)

    for video in train_videos:
        train_count += per_video_counts[video]

    for video in val_videos:
        val_count   += per_video_counts[video]

    for video in test_videos:
        test_count  += per_video_counts[video]

    fold_counts_map[kfold] = {"train": train_count, "val": val_count, "test": test_count}

#################################################################################################
################################### Save Per Test Fold Counts ###################################
#################################################################################################
with open(output_file_path, 'w') as f:
    for kfold in range(1, 6):
        # weight for each class is #negative_examples / #positive_examples

        train_weights        = fold_counts_map[kfold]["train"][1] / (fold_counts_map[kfold]["train"][0] + 1e-8)
        val_weights          = fold_counts_map[kfold]["val"][1]   / (fold_counts_map[kfold]["val"][0] + 1e-8)
        test_weights         = fold_counts_map[kfold]["test"][1]  / (fold_counts_map[kfold]["test"][0] + 1e-8)

        total_count_train    = fold_counts_map[kfold]["train"].sum(axis=0)[0]
        total_count_val      = fold_counts_map[kfold]["val"].sum(axis=0)[0]
        total_count_test     = fold_counts_map[kfold]["test"].sum(axis=0)[0]

        assert np.all(fold_counts_map[kfold]["train"].sum(axis=0) - total_count_train == 0), f"Fold {kfold}: Total counts don't match up!"
        assert np.all(fold_counts_map[kfold]["val"].sum(axis=0) - total_count_val == 0), f"Fold {kfold}: Total counts don't match up!"
        assert np.all(fold_counts_map[kfold]["test"].sum(axis=0) - total_count_test == 0), f"Fold {kfold}: Total counts don't match up!"

        print(f"\t * Fold {kfold}:")
        print(f"\t\t -> Train: #frames = {total_count_train:5d}")
        print(f"\t\t ->   Val: #frames = {total_count_val:5d}")
        print(f"\t\t ->  Test: #frames = {total_count_test:5d}")

        train_weight_list    = list(np.where(fold_counts_map[kfold]["train"][0] == 0, 1.0, train_weights))
        val_weight_list      = list(np.where(fold_counts_map[kfold]["val"][0] == 0, 1.0, val_weights))
        test_weight_list     = list(np.where(fold_counts_map[kfold]["test"][0] == 0, 1.0, test_weights))

        train_weight_string  = "".join([f", {count:.4f}" for count in train_weight_list])
        val_weight_string    = "".join([f", {count:.4f}" for count in val_weight_list])
        test_weight_string   = "".join([f", {count:.4f}" for count in test_weight_list])

        print(f"{kfold}{train_weight_string}", file=f)

print("")
print("".join(["#"] * 50))
print("\nDone!")