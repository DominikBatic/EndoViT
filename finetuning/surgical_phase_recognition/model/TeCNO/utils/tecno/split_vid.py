import skvideo.io
import numpy as np
import os
import glob
from tqdm import tqdm
from pathlib import Path
import configargparse

def build_configargparser(parser):
    parser.add("--data_root",
                               default="",
                               required=True,
                               type=str,
                               help='Path to the cholec80/videos folder with original .mp4 files.')

    parser.add("--img_size",
                               default=250,
                               type=int,
                               help='Output images will have the resolution [img_size, img_size]')

    parser.add("--fps",
                               default=25,
                               type=int,
                               help='Videos will be subsampled to this many FPS.')

    parser.add("--output_path",
                               default="",
                               required=True,
                               type=str,
                               help='Output images will be saved at output_path/cholec_split_{img_size}px_{fps}fps')

    known_args, _ = parser.parse_known_args()
    return parser, known_args


def videos_to_imgs(output_path="./datasets/cholec80",
                   input_path="./datasets/cholec80/videos",
                   pattern="*.mp4",
                   scale=250,
                   fps=25):
    output_path = Path(output_path) / f"cholec_split_{scale}px_{fps}fps"
    input_path = Path(input_path)

    dirs = list(input_path.glob(pattern))
    dirs.sort()
    output_path.mkdir(parents=True, exist_ok=True)

    for i, vid_path in enumerate(tqdm(dirs)):
        file_name = vid_path.stem
        out_folder = output_path / file_name
        # or .avi, .mpeg, whatever.
        out_folder.mkdir(exist_ok=True)
        os.system(
            f'ffmpeg -i {vid_path} -vf "scale={scale}:{scale},fps={fps}" {out_folder/file_name}_%06d.png'
        )
        print("Done extracting: {}".format(i + 1))


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, args = build_configargparser(parser)

    videos_to_imgs(output_path=args.output_path, input_path=args.data_root, scale=args.img_size, fps=args.fps)
