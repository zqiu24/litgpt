# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the OFT weights with the base model"""

from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import yaml

from litgpt.oft import GPT, Config, oft_filter, merge_oft_weights
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir


def merge_oft(
    checkpoint_dir: Path, pretrained_checkpoint_dir: Optional[Path] = None, precision: Optional[str] = None
) -> None:
    """Merges the OFT weights with the base model.

    See ``litgpt finetune oft``.

    Creates a new ``lit_model.pth`` file by merging the OFT weights (``lit_model.pth.oft``)
    with the original checkpoint weights.

    Arguments:
        checkpoint_dir: Path to the checkpoint directory with trained OFT weights, which is the output of
            ``litgpt finetune oft``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the OFT checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)
    pprint(locals())

    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth.oft")
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth").is_file():
        print("OFT weights have already been merged in this checkpoint.")
        return

    oft_params, meta_pretrained_checkpoint_dir, oft_precision = load_oft_metadata(checkpoint_dir)
    precision = precision if precision is not None else oft_precision

    if pretrained_checkpoint_dir is None:
        pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **oft_params)

    with fabric.init_module(), torch.device("meta"):
        model = GPT(config)
        # we don't care about these to perform merging
        model.cos = None
        model.sin = None

    oft_path = checkpoint_dir / "lit_model.pth.oft"
    pretrained_checkpoint = torch.load(str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True)
    oft_checkpoint = torch.load(str(oft_path), mmap=True)
    oft_checkpoint = oft_checkpoint.get("model", oft_checkpoint)

    # Merge OFT weights into the base model
    pretrained_checkpoint.update(oft_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    # since OFT finetuning only saves the OFT weights, we treat the oft weights dtype as the expected dtype
    oft_dtype = next(iter(oft_checkpoint.values())).dtype
    model.to(dtype=oft_dtype, device="cpu")
    merge_oft_weights(model)

    # Remove OFT parameters and the OFT linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not oft_filter(k, v)}
    save_path = checkpoint_dir / "lit_model.pth"
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")


def load_oft_metadata(checkpoint_dir: Path) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/oft.py` script."
        )

    with open(hparams_file, encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    oft_params = {k: v for k, v in hparams.items() if k.startswith("oft_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return oft_params, pretrained_checkpoint_dir, precision
