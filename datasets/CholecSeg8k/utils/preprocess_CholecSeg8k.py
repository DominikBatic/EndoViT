from pathlib import Path
import argparse
import torchvision.transforms.functional as F
from PIL import Image
import torch
import CholecSeg8k_color_dict_combined_classes as color_dict

def get_argument_parser():
    parser = argparse.ArgumentParser('Preprocess CholecSeg8k dataset', add_help=True)

    parser.add_argument('--output_dir', default='./datasets/CholecSeg8k_preprocessed',
                        help='Path where to save the output files.')
    
    parser.add_argument('--data_dir', default='./datasets/CholecSeg8k',
                        help="Path to CholecSeg8k dataset.")
    
    parser.add_argument('--already_processed_vids', default=[], nargs='+', type=int,
                        help= \
                        """If for some reason the processing stopped.\nYou can specify which videos have already been processed.\n\nNOTE: You should delete the last video subfolder that was left unfinished.""")
    
    return parser

def main(args):
    dataset = "CholecSeg8k"

    dataset_path = Path(args.data_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    print("".join(["*"] * 50 ))
    print(f"Preparing dataset ... {dataset}")
    print("".join(["*"] * 50 ))

    all_CholecSeg8k_vids = [1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37, 43, 48, 52, 55]

    videos = list(dataset_path.glob("*"))
    videos.sort()

    # remove already processed videos
    vids_to_remove = args.already_processed_vids

    if (vids_to_remove):
        indices_to_remove = []

        for i, vid in enumerate(videos):
            if(int(str(vid).split("/")[-1][-2:]) in vids_to_remove):
                indices_to_remove.append(i)
        
        indices_to_remove.reverse()

        for i in indices_to_remove:
            del videos[i]

        print(f"All CholecSeg8k videos: {all_CholecSeg8k_vids}")
        print(f"Removed: {vids_to_remove}")
        print(f"Will process only: {[int(str(vid).split('/')[-1][-2:]) for vid in videos]}", end="\n\n")

    print("Processing videos ...")

    for i, video in enumerate(videos):
        clips = list(video.glob("*"))
        clips.sort()

        for j, clip in enumerate(clips):
            frames = list(clip.glob("*_endo.png"))
            frames.sort()

            for k, frame in enumerate(frames):
                frame_path               = frame.relative_to(dataset_path)

                frame_dir                = frame_path.parent
                frame_path_base          = str(frame_path).split(".png")[0]

                frame_path_str           = frame_path_base + ".png"
                ground_truth_path_str    = frame_path_base + "_gt_mask.png"
                watershed_mask_path_str  = frame_path_base + "_watershed_mask.png"
                annotation_mask_path_str = frame_path_base + "_mask.png"
                color_mask_path_str      = frame_path_base + "_color_mask.png"

                open_img                = Image.open(dataset_path / frame_path_str)
                open_watershed_mask     = Image.open(dataset_path / watershed_mask_path_str)
                open_annotation_mask    = Image.open(dataset_path / annotation_mask_path_str)
                open_color_mask         = Image.open(dataset_path / color_mask_path_str)

                # Watershed and color masks of 4 images, namely:
                # "video18/video18_01139/frame_1216_endo.png",
                # "video35/video35_00780/frame_858_endo.png",
                # "video37/video37_00848/frame_865_endo.png",
                # "video37/video37_00848/frame_926_endo.png",

                # were color corrected by the authors. However, afterwards they were saved in RGBA instead of
                # RGB format like the rest of the dataset, this section corrects that
                if (           open_img.mode != 'RGB' or
                    open_watershed_mask.mode != 'RGB' or
                   open_annotation_mask.mode != 'RGB' or
                        open_color_mask.mode != 'RGB' ):
                    #print(f"{frame_path_str}, Mode: {open_img.mode} -- {open_watershed_mask.mode} -- {open_annotation_mask.mode} -- {open_color_mask.mode}")
                    with open(output_path / "incorrectly_saved_images.txt", mode="a") as err:
                        err.write(f"{frame_path_str}, Shapes: {open_img.mode} -- {open_watershed_mask.mode} -- {open_annotation_mask.mode} -- {open_color_mask.mode}\n")
                    
                    corrected_watershed_mask = Image.new("RGB", open_watershed_mask.size, (255, 255, 255))
                    corrected_watershed_mask.paste(open_watershed_mask, mask = open_watershed_mask.split()[3])
                    open_watershed_mask = corrected_watershed_mask

                    corrected_color_mask = Image.new("RGB", open_color_mask.size, (255, 255, 255))
                    corrected_color_mask.paste(open_color_mask, mask = open_color_mask.split()[3])
                    open_color_mask = corrected_color_mask

                open_img                 = F.to_tensor(open_img)
                open_watershed_mask      = F.to_tensor(open_watershed_mask)
                open_annotation_mask     = F.to_tensor(open_annotation_mask)
                open_color_mask          = F.to_tensor(open_color_mask)

                # create frame output_dir
                (output_path / frame_dir).mkdir(exist_ok=True, parents=True)

                # create ground truth mask
                open_gt_mask = torch.zeros(open_img.shape, dtype=torch.float)

                shape = open_gt_mask.shape

                open_gt_mask = open_gt_mask.reshape(3, -1)
                open_watershed_mask = open_watershed_mask.reshape(3, -1)

                for p in range(open_watershed_mask.shape[1]):
                    key = str(int(open_watershed_mask[0, p].item() * 255)).zfill(2)
                    open_gt_mask[:, p] = color_dict.watershed_to_class_v3[key]

                open_gt_mask = open_gt_mask.reshape(shape) / 255.
                open_watershed_mask = open_watershed_mask.reshape(shape)

                F.to_pil_image(open_img).save(output_path / frame_path_str)
                F.to_pil_image(open_gt_mask).save(output_path / ground_truth_path_str)
                F.to_pil_image(open_watershed_mask).save(output_path / watershed_mask_path_str)
                F.to_pil_image(open_annotation_mask).save(output_path / annotation_mask_path_str)
                F.to_pil_image(open_color_mask).save(output_path / color_mask_path_str)

        print(f"\t -> Video{str(video)[-2:]} done!")

    return

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)