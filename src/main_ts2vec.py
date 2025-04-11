import copy

import torch
import numpy as np
import random
from options import Options
from models import model_factory
from datasets.data_factory import data_provider


def main(config):

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    device = torch.device("cuda:0")

    # Build data
    train_dataset, train_loader = data_provider(config, "train")

    # Create model
    model_class = model_factory[config.model_name]
    model = model_class(config, 1, output_dims=config.output_dim, device=device)

    loss_log = model.fit(train_loader, n_epochs=config.epochs, verbose=True)
    model.save(
        f"{config.output_dir}/{config.data}_{config.output_dim}_{config.batch_size}.pt"
    )

    return loss_log


if __name__ == "__main__":
    args = Options().parse()  # `argparse` object
    origin_comment = args.comment
    for ii in range(args.itr):
        args_itr = copy.deepcopy(args)  # prevent itr forloop to change output_dir
        # config = setup(args_itr)  # save expriment files itr times
        main(args_itr)
