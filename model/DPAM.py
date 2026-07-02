
# The full code will be released upon acceptance of the manuscript.

import torch


def dummy_patch_weight(x):
    return torch.ones(x.shape[0], device=x.device)


def weighted_frame_loss(*args, **kwargs):
    return None


def update_frame_prediction_cache(*args, **kwargs):
    pass


def init_frame_prediction_cache():
    return {}, {}, {}


def frame_metrics(*args, **kwargs):
    return 0, 0, 0, 0, [], []