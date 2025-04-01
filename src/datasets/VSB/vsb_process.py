import pyarrow.parquet as pq
import pyarrow
import os
import pandas as pd
from multiprocessing import Pool
from argparse import ArgumentParser
import tqdm
import numpy as np
from pathlib import Path
from utils import dwt_single_signal_aligned_denoising
import matplotlib.pyplot as plt


def getArgparse():
    parser = ArgumentParser(description="load vsb data")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Choose the dataset type (train or test)",
    )
    parser.add_argument(
        "--phase-num",
        type=int,
        default=3,
        help="The phase num for each signal",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="The phase num for each signal",
    )
    parser.add_argument(
        "--n-proc",
        dest="process_num",
        type=int,
        default=8,
        help="The num of sub process",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=3046,
        help="random seed to generate the train and test dataset",
    )
    parser.add_argument(
        "--show-data", action="store_true", help="choose wether to show data"
    )
    return parser.parse_args()


def show_vsb_data(data, meta_data, args, reg_range=(-128, 127), start_id=0):
    """
    Show the VSB data
    :param data: data to be shown
    :param meta_data: meta data
    :param args: arguments
    """

    fig, axs = plt.subplots(3, 1, figsize=(15, 7))
    for i in range(3):
        axs[i].plot(data[start_id * 3 + i], color=plt.cm.tab10(i))
        axs[i].set_ylim(reg_range[0], reg_range[1])
        axs[i].axis("off")
    plt.show()


def preprocess_data(data, meta_data, args, batch_size=1000):
    """
    Preprocess data
    Attributes:
        data: data to be processed
        meta_data: meta data
        args: arguments
    """

    signal_ids = list(map(str, meta_data["signal_id"].values[args.start_id :]))
    num_signals = len(signal_ids)

    # Initialize pyarrow writer for Parquet
    parquet_writer = None
    df_denoised = pd.DataFrame()
    for i in range(0, num_signals, batch_size):
        if i + batch_size > num_signals:
            batch_size = num_signals - i
        batch_signal_ids = signal_ids[i : i + batch_size]
        all_denoised_signals_batch = []

        for signal_id in batch_signal_ids:
            single_signal = data[signal_id].values
            denoised_signal = dwt_single_signal_aligned_denoising(
                single_signal, aligned=True
            )
            # Convert denoised_signal to int8
            denoised_signal = denoised_signal.astype(np.int8)
            all_denoised_signals_batch.append(denoised_signal)

        all_denoised_signals_batch = np.array(all_denoised_signals_batch)
        df_denoised_batch = pd.DataFrame(all_denoised_signals_batch).T

        df_denoised = pd.concat([df_denoised, df_denoised_batch], axis=1)

        df_denoised.columns = range(len(df_denoised.columns))

    df_denoised.to_parquet(
        "./train_preprocess_aligned.parquet", engine="pyarrow", compression="snappy"
    )


def multiprocess_data(data, meta_data, args, process_func):
    """
    Multiprocess data processing
    :param data: data to be processed
    :param func: function to process data
    """
    signal_ids = meta_data["signal_id"].values[args.start_id :]
    if args.data_type == "test":
        # meta data cut into slices
        signal_ids = meta_data["signal_id"].values[args.start_id : -9]

    subprocess_num = 33
    if (len(signal_ids) / subprocess_num).is_integer():
        window_size = int(len(signal_ids) / subprocess_num)
    else:
        window_size = int(len(signal_ids) / 9)

    sub_ids = np.array_split(signal_ids, window_size)
    sub_ids = [list(map(str, chunk)) for chunk in sub_ids]

    save_path = Path.cwd()

    # multiprocessing
    with Pool(processes=args.process_num) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                process_func,
                [(chunk, save_path, data[chunk], args) for chunk in sub_ids],
            ),
            total=len(sub_ids),
            desc="Processing chunks",
        ):
            pass

    # process the last 9 signals
    if args.data_type == "test":
        test_ids = meta_data["signal_id"].values[-9:]
        test_ids = list(map(str, test_ids))
        process_func((test_ids, save_path, data[test_ids], args))


def main(meta_data, data, args):
    # set number of processes
    n_proc = 4

    # preprocess data,include high pass and denoise, the reshape to [2904,3,800000],and take an average of every 10 points to reduce the amount of data
    # preprocess data
    # preprocess_data(data, meta_data, args)

    # show data
    show_vsb_data(data, meta_data, args, start_id=152)


if __name__ == "__main__":

    args = getArgparse()
    # set data paths
    meta_data_train_path = r"E:\Graduate\projects\partial_discharge_monitoring_20230904\research\processing-paradigm\data\vsb-power-line-fault-detection\metadata_train.csv"
    df_train_path = r"E:\Graduate\projects\partial_discharge_monitoring_20230904\research\processing-paradigm\data\vsb-power-line-fault-detection\train.parquet"
    df_train_preprocess_path = r"E:\Graduate\projects\llm4pd_20250220\experiments\linearProjection\src\datasets\VSB\train_preprocess_aligned.parquet"

    # read data
    meta_data_train = pd.read_csv(meta_data_train_path)
    data_path = df_train_path if not args.show_data else df_train_preprocess_path
    df_train = pq.read_pandas(data_path).to_pandas()  # shape is [800000,8712]

    # process data for specific target
    main(meta_data_train, df_train, args)
