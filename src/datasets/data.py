import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class VSBClassification(BaseData):
    """
    Dataset class for vsb partial discharge dateset.
    Attributes:
        root_dir: str
            root directory of the dataset
        config: dict
            configuration dictionary

    """

    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.meta_data = pd.read_csv(config["meta_dir"])
        self.d_piece = config["vsb_piece_dim"]
        self.config = config

        self.feature_df, self.labels = self.load_data(root_dir)

    def load_data(self, filepath):
        """
        Load data
        Attributes:
            filepath: str
                path to the data file
        Returns:
            df: pd.DataFrame
                data frame of the vsb data,shape is [80000,8712]
            labels: list
                list of labels, shape is [8712,]
        """
        df = pq.read_pandas(filepath).to_pandas()  # data shape is [800000,8712]

        measurement_ids = self.meta_data["id_measurement"]
        self.class_names = self.meta_data["target"].unique()
        labels = pd.DataFrame(self.get_labels(self.meta_data))

        # down-sample data to [80000,8712]
        data = df.values
        n_rows, n_cols = data.shape
        downsampled_rows = n_rows // 10

        reshaped_data = data[: downsampled_rows * 10].reshape(-1, 10, n_cols)
        max_indices = np.abs(reshaped_data).argmax(axis=1)
        downsampled_data = reshaped_data[
            np.arange(downsampled_rows)[:, None], max_indices, np.arange(n_cols)
        ]
        df = pd.DataFrame(downsampled_data, columns=df.columns)
        
        # transposition, for index setting
        df = df.T

        self.d_fea = df.shape[1]
        self.n_piece = self.d_fea // self.d_piece
        self.n_phase = len(self.meta_data["phase"].unique())
        self.max_seq_len = self.n_piece * self.n_phase  # 3 *160 = 480

        # reset data index for easy access
        df = df.reset_index(drop=True).set_index(measurement_ids)

        return df, labels

    def get_labels(self, meta_data_train):
        """
        Get labels from meta data
        assume that 3 phases as a measurement
        """
        signal_ids = meta_data_train["id_measurement"].unique()
        labels = labels = [
            (
                1
                if (
                    meta_data_train[meta_data_train["id_measurement"] == id][
                        "target"
                    ].values
                    == 1
                ).sum()
                >= 1
                else 0
            )
            for id in signal_ids
        ]

        return labels


data_factory = {"VSB": VSBClassification}
