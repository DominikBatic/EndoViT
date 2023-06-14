import subprocess
import glob
from pathlib import Path

import argparse
from PIL import Image
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('Copy some part of cholec80.', add_help=False)

    parser.add_argument('--cholec80_path', default='./datasets/Cholec80',
                        help='Path to cholec80.')

    parser.add_argument('--output_dir', default='./datasets',
                        help='Path where to save the output.')
    
    parser.add_argument('--split', default='frames',
                        help='Name of the folder inside cholec80_path where to look for pre-processed cholec80 videos.')

    return parser


def main(args):
    cholec80_path = Path(args.cholec80_path).resolve()
    cholec80_folder_path = cholec80_path / args.split

    create_dictionary = {
        "Cholec80_for_Segmentation": set([i for i in range(1, 81)]) - set([12, 17, 27, 52]),
        "Cholec80_for_ActionTripletDetection": set([i for i in range(1, 81)]) - set([78, 43, 62, 35, 74,  1, 56,  4, 13,]),
        "Cholec80_for_SurgicalPhaseRecognition": set([i for i in range(1, 41)]),
        "Cholec80_for_Validation": set([3, 24, 40, 16, 32, 34, 7, 14, 29, 8, 31, 37, 15, 30, 18, 11, 6, 2, 33, 21, 23, 25, 19, 20])
    }

    imgs_per_folder = {}

    for out_folder_name, indices_to_take in create_dictionary.items():
        output_path = Path(args.output_dir)
        output_path = output_path / "validation_dataset" if out_folder_name == "Cholec80_for_Validation" else output_path / "Endo700k"
        out_folder = output_path / out_folder_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        image_counter = 0
        
        videos = [x.resolve() for x in cholec80_folder_path.iterdir() if int(str(x)[-2:]) in indices_to_take]
        videos.sort()

        print("Preparing ... " + out_folder_name)
        for video_path in tqdm(videos, total=len(videos)):
            print("\t -> Processing ... " + "video" + (str(video_path).split("/")[-1][-2:]), end="\n")

            # each folder contains images saved as (.png)
            imgs = list(video_path.glob("*.png"))

            # copy images to output folder
            for img in tqdm(imgs, total=len(imgs)):
                image_counter += 1

                source = str(img)
                destination = str(out_folder / str(img.relative_to(cholec80_path)).replace("/", "__"))

                subprocess.run(["cp", f"{source}", f"{destination}"])

        imgs_per_folder[out_folder_name] = image_counter

    print("")
    print("*" * 50)
    print("*" * 50)
    print("")
    print("Summary:")
    print("-" * 10)

    # print #images for each folder
    total = 0

    for key, value in imgs_per_folder.items():
        print("{:>40}: {:>10}".format(key, value))

    print("")
    print("*" * 50)
    print("*" * 50)
    print("")

    return


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
