
import argparse
import datetime
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from fvcore.common.timer import Timer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import *
from models import *


@hydra.main(config_path="./configs", config_name="default_sub4.yaml")
def main(cfg: DictConfig) -> None:

    if "experiments" in cfg.keys():
        cfg = OmegaConf.merge(cfg, cfg.experiments)

    if "debug" in cfg.keys():
        logger.info(f"Run script in debug")
        cfg = OmegaConf.merge(cfg, cfg.debug)

    # A logger for this file
    logger = logging.getLogger(__name__)

    # NOTE: hydra causes the python file to run in hydra.run.dir by default
    logger.info(f"Run script in {HydraConfig.get().run.dir}")

    assert cfg.test.checkpoint_model != "", "Specify path to checkpoint model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (cfg.test.channels, cfg.test.image_height, cfg.test.image_width)

    # NOTE: With hydra, the python file runs in hydra.run.dir by default, so set the dataset path to a full path or an appropriate relative path
    dataset_path = Path(cfg.dataset.root) / cfg.dataset.frames
    split_path = Path(cfg.dataset.root) / cfg.dataset.split_file
    assert dataset_path.exists(), "Video image folder not found"
    assert (
        split_path.exists()
    ), "The file that describes the split of train/test not found."

    # Define test set
    test_dataset = Dataset(
        dataset_path=dataset_path,
        split_path=split_path,
        split_number=cfg.dataset.split_number,
        input_shape=image_shape,
        sequence_length=cfg.test.sequence_length,
        training=False,
    )

    # Define test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
    )

    # Classification criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Define network
    model = CNNLSTM(
        num_classes=cfg.test.num_classes,
        latent_dim=cfg.test.latent_dim,
        lstm_layers=cfg.test.lstm_layers,
        hidden_dim=cfg.test.hidden_dim,
        bidirectional=cfg.test.bidirectional,
        attention=cfg.test.attention,
    )
    ckpt = torch.load(cfg.test.checkpoint_model, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    (_x, _y) = next(iter(test_dataloader))
    _x = _x.to(device)
    
    logger.info("exporting the model..")
    torch.onnx.export(
        model,
        _x,
        "tmk_action.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

if __name__ == "__main__":
    main()
