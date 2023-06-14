# Main script to run the training.

import argparse
import json
from src.trainer import Trainer
from src import util
import wandb
from datetime import datetime
from pathlib import Path
import os
from contextlib import redirect_stdout, redirect_stderr

import torch
import numpy as np
import torch.backends.cudnn as cudnn

def get_config():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-c", type=str, default="", help="Path to the \".json\" config file of the project.", required=True)

    args, unparsed = parser.parse_known_args()

    config_path = Path(args.c)
    assert config_path.is_file(), "Please provide a valid path to a \".json\" config file."

    with open(config_path, "r") as read_file:
        config = json.load(read_file)   
        return config

def main():
    config = get_config()

    # Set up output directory
    out_dir = Path(config["Project Setup"]["output_dir"]).resolve()

    # Automatically count which run it is.
    # Note: The output files go to directory named as 'output_dir/run_{run_number}_{date_and_time}_{run_name}'.
    #       The following code will look at all subfolders in 'output_dir', determine the last {run_number} and
    #       increase it by 1.

    current_run_number = "1"
    runs = [directory.name for directory in out_dir.iterdir()]
    if (runs):
        current_run_number = str(max(map(lambda dir_name : int(dir_name.split("_")[1]) + 1 if dir_name.split("_")[0] == "run" else 1, runs)))

    date_str = datetime.now().strftime("%H:%M-%d.%m.%y")
    run_name = "run_{}_{}__{}".format(current_run_number.zfill(4), date_str, config['Project Setup']['run_name'])
    out_dir = out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config['Project Setup']['output_dir'] = str(out_dir)

    log_dir = out_dir / config["Project Setup"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Redirecting std_out and std_err to the output folder.
    stdout_file_path = out_dir / 'out.txt'
    stderr_file_path = out_dir / 'err.txt'

    print("Redirected stdout to {}".format(stdout_file_path))
    print("Redirected stderr to {}".format(stderr_file_path))

    with open(stderr_file_path, 'w') as f_err, redirect_stderr(f_err):    
        with open(stdout_file_path, 'w') as f_out, redirect_stdout(f_out):
            print("Running ...")
            print(f"\t -> Project: \"{config['Project Setup']['project_name']}\"")
            print(f"\t\t -> Run: \"{config['Project Setup']['run_name']}\"")
            print(f"\t\t -> Job dir: {Path(__file__).resolve().parent}")
            print(f"\t\t -> Output dir: {out_dir}")

            # Print out "config" file.
            util.config_summary(config)
            
            # Set up WandB and seed.
            util.header("Initialization")

            # WandB
            if(config["Loggers"]['WandB']['enable']):
                print("Setting up WandB ... ", end="")
                wandb.login()
                wandb.init(
                    project=config['Project Setup']['project_name'],
                    name=run_name,
                    tags=config["Loggers"]['WandB']['tags'],
                    dir=str(out_dir / config["Project Setup"]["log_dir"])
                )
                wandb.config.update(config)
            else:
                print("Turning off WandB ... ", end="")
                os.environ["WANDB_MODE"]="offline"

            print("Done!", end="")

            util.new_section()

            # Fix the seed for reproducibility.
            seed = config['General Hyperparams']['seed']

            if (seed != -1):
                print("Ensuring reproducibility ...")
                print(f"\t -> torch.manual_seed: {seed}")
                print(f"\t ->    np.random.seed: {seed}")
                print(f"\t ->     cudnn.enabled: True")
                print(f"\t ->   cudnn.benchmark: True")

                torch.manual_seed(seed)
                np.random.seed(seed)
                cudnn.enabled = True
                cudnn.benchmark = True
            else:
                print("Reproducibility DISABLED ...")

            trainer = Trainer(config)

            util.new_line(times=4)
            util.header("Training", separator="#")
            best_result_dict = trainer.run_training()

            util.new_line(times=4)
            util.header("Testing", separator="#")
            trainer.run_testing(best_result_dict["ckpt_path"])

            wandb.finish()

if __name__ == "__main__":
    main()