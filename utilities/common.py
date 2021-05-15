from typing import Optional

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_callback(callbacks) -> Optional[ModelCheckpoint]:
    try:
        return next((c for c in callbacks if type(c) == ModelCheckpoint))
    except StopIteration:
        return None
