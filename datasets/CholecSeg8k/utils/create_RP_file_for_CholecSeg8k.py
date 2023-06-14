import subprocess
import os
import glob
from pathlib import Path
import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser('Create CholecSeg8k Relative Paths (RP) file', add_help=True)

    parser.add_argument('--output_dir', default='./datasets/CholecSeg8k_preprocessed',
                        help='Path where to save the output files.')
    
    parser.add_argument('--data_dir', default='./datasets/CholecSeg8k_preprocessed',
                        help="Path to CholecSeg8k dataset.")
    
    return parser

def main(args):
    dataset = "CholecSeg8k"

    dataset_path = Path(args.data_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    videos = list(dataset_path.glob("*"))
    videos.sort()

    print("Processing videos ...")

    output_file_path = str(output_path / "RP_CholecSeg8k.csv")

    with open(output_file_path, 'w') as f:
        # header
        print("image,gt_segmentation_mask,watershed_mask,annotation_mask,color_mask", file=f)

        for i, video in enumerate(videos):
            clips = list(video.glob("*"))
            clips.sort()

            for j, clip in enumerate(clips):
                frames = list(clip.glob("*_endo.png"))
                frames.sort()

                for k, frame in enumerate(frames):
                    frame_path              = frame.relative_to(dataset_path)

                    frame_path_base         = str(frame_path).split(".png")[0]

                    frame_path_str           = frame_path_base + ".png"
                    ground_truth_path_str    = frame_path_base + "_gt_mask.png"
                    watershed_mask_path_str  = frame_path_base + "_watershed_mask.png"
                    annotation_mask_path_str = frame_path_base + "_mask.png"
                    color_mask_path_str      = frame_path_base + "_color_mask.png"

                    print(",".join([
                        frame_path_str,
                        ground_truth_path_str,
                        watershed_mask_path_str,
                        annotation_mask_path_str,
                        color_mask_path_str
                        ]), file=f)

            print(f"\t -> Video{str(video)[-2:]} done!")

    return

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)