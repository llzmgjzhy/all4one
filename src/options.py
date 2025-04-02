import argparse


class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description="Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments."
        )

        # basic config
        self.parser.add_argument(
            "--task",
            choices={
                "imputation",
                "transduction",
                "classification",
                "regression",
                "forecast",
            },
            default="forecast",
            help=(
                "Training objective/task: imputation of masked values,\n"
                "                          transduction of features to other features,\n"
                "                          classification of entire time series,\n"
                "                          regression of scalar(s) for entire time series"
                "                          forecasting of future values"
            ),
        )
        self.parser.add_argument(
            "--output_dir",
            default="./output",
            help="Root output directory. Must exist. Time-stamped directories will be created inside.",
        )
        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--comment",
            type=str,
            default="",
            help="A comment/description of the experiment",
        )
        self.parser.add_argument(
            "--no_timestamp",
            action="store_true",
            help="If set, a timestamp will not be appended to the output directory name",
        )
        self.parser.add_argument(
            "--records_file",
            default="./records.xls",
            help="Excel file keeping all records of experiments",
        )

        # data loader
        self.parser.add_argument(
            "--root_path",
            type=str,
            default="./dataset",
            help="root path of the data file",
        )
        self.parser.add_argument(
            "--data_path", type=str, default="ETTh1.csv", help="data file"
        )
        self.parser.add_argument(
            "--data", type=str, required=True, default="ETTm1", help="dataset type"
        )
        self.parser.add_argument(
            "--features",
            type=str,
            default="M",
            help="forecasting task, options:[M, S, MS]; "
            "M:multivariate predict multivariate, S: univariate predict univariate, "
            "MS:multivariate predict univariate",
        )
        self.parser.add_argument(
            "--freq",
            type=str,
            default="h",
            help="freq for time features encoding, "
            "options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], "
            "you can also use more detailed freq like 15min or 3h",
        )
        self.parser.add_argument(
            "--target", type=str, default="OT", help="target feature in S or MS task"
        )

        # forecasting task
        self.parser.add_argument(
            "--seq_len", type=int, default=512, help="input sequence length"
        )
        self.parser.add_argument(
            "--label_len", type=int, default=48, help="start token length"
        )
        self.parser.add_argument(
            "--pred_len", type=int, default=96, help="prediction sequence length"
        )
        self.parser.add_argument(
            "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
        )

        # System
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=1,
            help="Print batch info every this many batches",
        )
        self.parser.add_argument(
            "--gpu", type=str, default="0", help="GPU index, -1 for CPU"
        )
        self.parser.add_argument(
            "--n_proc",
            type=int,
            default=-1,
            help="Number of processes for data loading/preprocessing. By default, equals num. of available cores.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=10,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            help="Seed used for splitting sets. None by default, set to an integer for reproducibility",
        )

        # Training process
        self.parser.add_argument(
            "--epochs", type=int, default=50, help="Number of training epochs"
        )
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=2,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )
        self.parser.add_argument(
            "--optimizer", choices={"Adam", "RAdam"}, default="Adam", help="Optimizer"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate (default holds for batch size 64)",
        )
        self.parser.add_argument(
            "--lr_step",
            type=str,
            default="1000000",
            help="Comma separated string of epochs when to reduce learning rate by a factor of 10."
            " The default is a large value, meaning that the learning rate will not change.",
        )
        self.parser.add_argument(
            "--lr_factor",
            type=str,
            default="0.1",
            help=(
                "Comma separated string of multiplicative factors to be applied to lr "
                "at corresponding steps specified in `lr_step`. If a single value is provided, "
                "it will be replicated to match the number of steps in `lr_step`."
            ),
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=64, help="Training batch size"
        )
        self.parser.add_argument(
            "--key_metric",
            choices={"loss", "accuracy", "precision", "mcc", "mse_loss"},
            default="loss",
            help="Metric used for defining best epoch",
        )
        self.parser.add_argument(
            "--freeze",
            action="store_true",
            help="If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer",
        )
        self.parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )

        # Model
        self.parser.add_argument("--is_gpt", type=int, default=1)
        self.parser.add_argument("--pretrain", type=int, default=1)
        self.parser.add_argument(
            "--model_name",
            default="GPT4TS",
            help="Model class",
        )
        self.parser.add_argument(
            "--max_seq_len",
            type=int,
            help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""",
        )
        self.parser.add_argument(
            "--patch_size", type=int, default=64, help="patch_size"
        )
        self.parser.add_argument("--stride", type=int, default=64, help="stride")
        self.parser.add_argument(
            "--d_model",
            type=int,
            default=64,
            help="Internal dimension of transformer embeddings",
        )
        self.parser.add_argument(
            "--num_layers",
            type=int,
            default=3,
            help="Number of transformer encoder layers (blocks)",
        )
        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout applied to most transformer encoder layers",
        )
        self.parser.add_argument(
            "--pos_encoding",
            choices={"fixed", "learnable"},
            default="fixed",
            help="Internal dimension of transformer embeddings",
        )
        self.parser.add_argument(
            "--activation",
            choices={"relu", "gelu"},
            default="gelu",
            help="Activation to be used in transformer encoder",
        )
        self.parser.add_argument(
            "--normalization_layer",
            choices={"BatchNorm", "LayerNorm"},
            default="BatchNorm",
            help="Normalization layer to be used internally in transformer encoder",
        )
        self.parser.add_argument(
            "--split_num",
            type=int,
            default=5,
            help="Dropout applied to most transformer encoder layers",
        )
        self.parser.add_argument(
            "--loss",
            choices={"cross_entropy", "focal", "mse"},
            default="cross_entropy",
            help="loss used for train model",
        )
        self.parser.add_argument(
            "--d_ff",
            type=int,
            default=32,
            help="dimension of fcn",
        )
        self.parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        self.parser.add_argument(
            "--llm_dim", type=int, default="768", help="LLM model dimension"
        )  # LLama7b:4096; GPT2-small:768; BERT-base:768
        self.parser.add_argument(
            "--prompt_domain",
            type=int,
            default=None,
            help="if set, will use the specified domain for prompt, otherwise not use prompt and subsequent embedding concat",
        )
        self.parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        self.parser.add_argument(
            "--enc_in", type=int, default=7, help="encoder input size"
        )
        self.parser.add_argument(
            "--dec_in", type=int, default=7, help="decoder input size"
        )
        self.parser.add_argument(
            "--pct_start", type=float, default=0.2, help="pct_start"
        )
        self.parser.add_argument("--c_out", type=int, default=7, help="output size")
        self.parser.add_argument("--llm_layers", type=int, default=6)
        self.parser.add_argument("--percent", type=int, default=100)

    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(",")]
        args.lr_factor = [float(i) for i in args.lr_factor.split(",")]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor
        ), "You must specify as many values in `lr_step` as in `lr_factors`"

        return args
