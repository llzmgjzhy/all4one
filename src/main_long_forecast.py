import logging
import os
import sys
import time
import copy

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import random
from accelerate import Accelerator

from options import Options
from running import setup, pipeline_factory, validate, test, NEG_METRICS
from utils import utils
from optimizers import get_optimizer
from models import model_factory
from models.loss import get_loss_module
from datasets.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, load_content


def main(config):
    total_epoch_time = 0
    total_start_time = time.time()
    accelerator = Accelerator()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config.output_dir, "output.log"))
    logger.addHandler(file_handler)

    logger.info(f"Running:\n{' '.join(sys.argv)}\n")

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    device = torch.device("cuda:0")
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"Device index: {torch.cuda.current_device()}")

    # load content
    config.content = load_content(config)

    # Build data
    logger.info("Loading and preprocessing data ...")
    train_dataset, train_loader = data_provider(config, "train")
    val_dataset, val_loader = data_provider(config, "val")
    test_dataset, test_loader = data_provider(config, "test")

    # Create model
    logger.info("Creating model ...")
    model_class = model_factory[config.model_name]
    model = model_class(config, device)

    logger.info(f"Model:\n{model}")
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=config.patience)

    # initialize the optimizer
    optim_class = get_optimizer(config.optimizer)
    optimizer = optim_class(model.parameters(), lr=config.lr)

    # loss criterion
    loss_module = get_loss_module(config)

    # initialize the scheduler
    if config.lradj == "COS":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=config.pct_start,
            epochs=config.epochs,
            max_lr=config.lr,
        )

    train_loader, val_loader, test_loader, model, optimizer, scheduler = (
        accelerator.prepare(
            train_loader, val_loader, test_loader, model, optimizer, scheduler
        )
    )

    start_epoch = 0

    # initialize runner, responsible for training, validation and testing
    runner_class = pipeline_factory(config)

    trainer = runner_class(
        model,
        train_loader,
        device,
        loss_module,
        config,
        optimizer=optimizer,
        accelerator=accelerator,
        print_interval=config.print_interval,
        console=config.console,
    )
    val_evaluator = runner_class(
        model,
        val_loader,
        device,
        loss_module,
        config,
        accelerator=accelerator,
        print_interval=config.print_interval,
        console=config.console,
    )

    tensorboard_writer = SummaryWriter(config.tensorboard_dir)

    best_value = (
        1e16 if config.key_metric in NEG_METRICS else -1e16
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

    # training
    # fmt: off
    for epoch in tqdm(range(start_epoch + 1, config.epochs + 1), desc="Training Epoch", leave=False):
    # fmt: on
        # mark = epoch if config["save_all"] else "last"
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(
            epoch
        )  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        accelerator.print()
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
            (epoch == config.epochs)
            or (epoch == start_epoch + 1)
            or (epoch % config.val_interval == 0)
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
            early_stopping(best_value)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

        # Learning rate scheduling
        if config.lradj == "COS":
            scheduler.step()
            accelerator.print("lr = {:.10f}".format(optimizer.param_groups[0]["lr"]))
        else:
            adjust_learning_rate(optimizer, epoch + 1, config)


    # testing
    model.load_state_dict(torch.load(os.path.join(config.save_dir, "model_best.pth"))["state_dict"])
    test_evaluator = runner_class(
        model,
        test_loader,
        device,
        loss_module,
        config,
        accelerator=accelerator,
        print_interval=config.print_interval,
        console=config.console,
    )
    aggr_metrics_test = test(test_evaluator)

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(
        config.output_dir, "metrics_" + config.experiment_name + ".xlsx"
    )
    book = utils.export_performance_metrics(
        metrics_filepath, metrics, header, sheet_name="metrics"
    )

    # Export record metrics to a file accumulating records from all experiments
    utils.register_test_record(
        config.records_file,
        config.initial_timestamp,
        config.experiment_name,
        best_metrics,
        aggr_metrics_val,
        aggr_metrics_test,
        comment=config.comment,
    )

    logger.info(
        "Best {} was {}. Other metrics: {}".format(
            config.key_metric, best_value, best_metrics
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
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading packages ...")

    args = Options().parse()  # `argparse` object
    origin_comment = args.comment
    for ii in range(args.itr):
        args_itr = copy.deepcopy(args)  # prevent itr forloop to change output_dir
        config = setup(args_itr)  # save expriment files itr times
        config.comment = origin_comment + f" itr{ii}"
        main(config)
