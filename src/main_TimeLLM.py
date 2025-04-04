import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import os
import sys
import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from options import Options
from running import setup, pipeline_factory, check_progress, validate, NEG_METRICS
from utils import utils
from optimizers import get_optimizer
from models import TimeLLM
from models.loss import get_loss_module
from datasets.data import data_factory


def main(config):

    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)

    logger.info(f"Running:\n{' '.join(sys.argv)}\n")

    if config["seed"] is not None:
        torch.manual_seed(config["seed"])

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["gpu"] != -1) else "cpu"
    )
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"Device index: {torch.cuda.current_device()}")

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(config["data_dir"], config)

    # spilt data
    splits = StratifiedKFold(
        n_splits=config["split_num"], shuffle=True, random_state=config["seed"]
    ).split(my_data.meta_data["id_measurement"].unique(), my_data.labels)

    logger.info(
        f"train data is divided into {config['split_num']} splits, and will take cross validation"
    )

    # Create model
    logger.info("Creating model ...")
    if config["text_model"] == "GPT2" or config["text_model"] == "LLAMA":
        model = TimeLLM.Model(config, my_data).float()

    if config["freeze"]:
        for name, param in model.named_parameters():
            if name.startswith("output_layer"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info(f"Model:\n{model}")
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    # initialize the optimizer

    if config["global_reg"]:
        weight_decay = config["l2_reg"]
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config["l2_reg"]

    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(
        model.parameters(), lr=config["lr"], weight_decay=weight_decay
    )

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config["lr"]  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(
            model,
            config["load_model"],
            optimizer,
            config["resume"],
            config["change_output"],
            config["lr"],
            config["lr_step"],
            config["lr_factor"],
        )
    model.to(device)

    loss_module = get_loss_module(config)

    # initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)

    for idx, (train_indices, val_indices) in enumerate(splits):
        val_dataset = dataset_class(my_data, val_indices)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
        )

        train_dataset = dataset_class(my_data, train_indices)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
        )

        trainer = runner_class(
            model,
            train_loader,
            device,
            loss_module,
            optimizer,
            l2_reg=output_reg,
            print_interval=config["print_interval"],
            console=config["console"],
        )
        val_evaluator = runner_class(
            model,
            val_loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )

        tensorboard_writer = SummaryWriter(config["tensorboard_dir"])

        best_value = (
            1e16 if config["key_metric"] in NEG_METRICS else -1e16
        )  # initialize with +inf or -inf depending on key metric
        metrics = (
            []
        )  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
        best_metrics = {}

        # Evaluate on validation before training
        aggr_metrics_val, best_metrics, best_value = validate(
            val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch=0
        )
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        logger.info("Starting training...")

        for epoch in tqdm(
            range(start_epoch + 1, config["epochs"] + 1),
            desc="Training Epoch",
            leave=False,
        ):

            mark = epoch if config["save_all"] else "last"
            epoch_start_time = time.time()
            aggr_metrics_train = trainer.train_epoch(
                epoch
            )  # dictionary of aggregate epoch metrics
            epoch_runtime = time.time() - epoch_start_time
            print()
            print_str = f"Epoch {epoch} Training Summary: "
            for k, v in aggr_metrics_train.items():
                tensorboard_writer.add_scalar(f"{k}/train", v, epoch)
                print_str += f"{k}: {v:8f} | "

            logger.info(print_str)
            logger.info(
                "Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
                    *utils.readable_time(epoch_runtime)
                )
            )
            total_epoch_time += epoch_runtime
            avg_epoch_time = total_epoch_time / (epoch - start_epoch)
            avg_batch_time = avg_epoch_time / len(train_loader)
            avg_sample_time = avg_epoch_time / len(train_dataset)
            logger.info(
                "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(
                    *utils.readable_time(avg_epoch_time)
                )
            )

            logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
            logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

            # evaluate if first or last epoch or at specified interval
            if (
                (epoch == config["epochs"])
                or (epoch == start_epoch + 1)
                or (epoch % config["val_interval"] == 0)
            ):
                aggr_metrics_val, best_metrics, best_value = validate(
                    val_evaluator,
                    tensorboard_writer,
                    config,
                    best_metrics,
                    best_value,
                    epoch,
                )
                metrics_names, metrics_values = zip(*aggr_metrics_val.items())
                metrics.append(list(metrics_values))

            utils.save_model(
                os.path.join(config["save_dir"], "model_{}.pth".format(mark)),
                epoch,
                model,
                optimizer,
            )

            # Learning rate scheduling
            if epoch == config["lr_step"][lr_step]:
                utils.save_model(
                    os.path.join(config["save_dir"], "model_{}.pth".format(epoch)),
                    epoch,
                    model,
                    optimizer,
                )
                lr = lr * config["lr_factor"][lr_step]
                if (
                    lr_step < len(config["lr_step"]) - 1
                ):  # so that this index does not get out of bounds
                    lr_step += 1
                logger.info("Learning rate updated to: ", lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Difficulty scheduling
            if config["harden"] and check_progress(epoch):
                train_loader.dataset.update()
                val_loader.dataset.update()

        # Export evolution of metrics over epochs
        header = metrics_names
        metrics_filepath = os.path.join(
            config["output_dir"], "metrics_" + config["experiment_name"] + ".xls"
        )
        book = utils.export_performance_metrics(
            metrics_filepath, metrics, header, sheet_name="metrics"
        )

        # Export record metrics to a file accumulating records from all experiments
        utils.register_record(
            config["records_file"],
            config["initial_timestamp"],
            config["experiment_name"],
            best_metrics,
            aggr_metrics_val,
            comment=config["comment"],
        )

        logger.info(
            "Best {} was {}. Other metrics: {}".format(
                config["key_metric"], best_value, best_metrics
            )
        )
        logger.info("All Done!")

        total_runtime = time.time() - total_start_time
        logger.info(
            "Total runtime: {} hours, {} minutes, {} seconds\n".format(
                *utils.readable_time(total_runtime)
            )
        )

        return best_value


if __name__ == "__main__":

    args = Options().parse()  # `argparse` object
    config = setup(args)
    main(config)
