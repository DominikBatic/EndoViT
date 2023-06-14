import configargparse


def build_configargparser(parser):
    model_group = parser.add_argument_group(title='Model options')
    dataset_group = parser.add_argument_group(title='Dataset options')
    module_group = parser.add_argument_group(title='Module options')
    trainer_group = parser.add_argument_group(title='Trainer options')

    # gpu args
    trainer_group.add_argument("--gpus",
                               type=int,
                               #nargs='+',
                               default=0,
                               help="how many gpus / -1 means all")
    trainer_group.add_argument(
        "--accelerator",
        type=str,
        default="ddp",
        help="supports four options dp, ddp, ddp_spawn, ddp2",
    )

    trainer_group.add_argument("--resume_from_checkpoint",
                               type=str,
                               default=None)

    trainer_group.add_argument("--log_every_n_steps",
                               type=int,
                               default=50)

    # config module e.g. classification_multiclass
    module_group.add_argument("--module", type=str, required=True)

    # config model e.g. densenet.DenseNet121
    model_group.add_argument("--model", type=str, required=True)

    # config dataset e.g. ham10k.HAM10k
    dataset_group.add_argument("--data_root",
                               default="",
                               required=True,
                               type=str)
    dataset_group.add_argument("--dataset", type=str, required=True)
    dataset_group.add_argument("--out_features", type=int, required=True)
    dataset_group.add_argument("--train_percent_check",
                               type=float,
                               default=1.0)
    dataset_group.add_argument("--val_percent_check", default=1.0, type=float)
    dataset_group.add_argument("--test_percent_check", default=1.0, type=float)
    dataset_group.add_argument("--overfit_pct", default=0.0, type=float)

    # config trainer
    trainer_group.add_argument("--log_interval", type=int, default=100)
    trainer_group.add_argument("--num_workers", type=int, default=8)

    # log root
    trainer_group.add_argument("--num_sanity_val_steps", default=5, type=int)

    # uses this much of dataset

    # max and min epoch
    trainer_group.add_argument("--max_epochs", default=1000, type=int)
    trainer_group.add_argument("--min_epochs", default=1, type=int)

    # check logging frequency
    trainer_group.add_argument("--check_val_every_n_epoch",
                               default=1,
                               type=int)
    trainer_group.add_argument("--save_top_k", default=1, type=int)  #-1 == all
    trainer_group.add_argument("--early_stopping_metric",
                               type=str,
                               default="val_loss")
    # logging
    trainer_group.add_argument("--log_save_interval", default=100, type=int)
    trainer_group.add_argument("--row_log_interval", default=100, type=int)

    trainer_group.add_argument("--fast_dev_run", default=False, type=str)
    trainer_group.add_argument("--name", default=None, type=str)
    dataset_group.add_argument("--input_height", default=224, type=int)
    dataset_group.add_argument("--input_width", default=224, type=int)
    trainer_group.add_argument("--on_polyaxon", action="store_true")

    trainer_group.add_argument("--output_path", type=str, default="logs")

    # ADDED
    trainer_group.add_argument("--seed", type=int, default=1665)
    trainer_group.add_argument('--data_efficient_training', type=int, nargs='*', default=[], 
                            help=
                            """
                            To test out how our models perform when less training data is available we allow manually selecting a subset of training videos to take
                            as the training dataset. In our tests we will select either 2, 4 or 8 videos. The videos should be specified as a list of integers. 
                            
                            NOTE 1: If you select a video number that is outside the training dataset an error will be raised. 
                            NOTE 2: Leave this list empty to use the full training dataset.
                            """)
    trainer_group.add_argument('--wandb_project_name', type=str, default="TeCNO_FINAL",  help='Name of the wandb project where to log the run.')
    trainer_group.add_argument('--wandb_name_suffix', type=str, default="", help='Suffix to attach to the name of the run in wandb.')
    trainer_group.add_argument('--wandb_tags', type=str, nargs='*', default=["TeCNO"],  help='A list of wandb tags for the run.')

    # END_ADDED

    known_args, _ = parser.parse_known_args()
    return parser, known_args
