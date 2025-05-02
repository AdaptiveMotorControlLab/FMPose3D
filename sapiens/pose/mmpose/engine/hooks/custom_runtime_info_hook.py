# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Union

import torch
import numpy as np

from mmengine.hooks import RuntimeInfoHook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


def _is_tensor_safe_for_scalar(value: Any) -> bool:
    """Check if tensor can be safely converted to scalar."""
    if isinstance(value, torch.Tensor):
        # For tensors with multiple elements, we'll use mean to convert to scalar
        return True
    elif isinstance(value, np.ndarray):
        return True
    elif isinstance(value, (int, float, np.number)):
        return True
    return False


def _make_scalar_safe(value: Any) -> Union[int, float]:
    """Convert tensor to scalar safely, using mean for multi-element tensors."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            # Take mean for multi-element tensors
            return value.mean().item()
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.mean().item()
    elif isinstance(value, (int, float, np.number)):
        return float(value)
    raise ValueError(f"Cannot convert {type(value)} to scalar")


@HOOKS.register_module()
class CustomRuntimeInfoHook(RuntimeInfoHook):
    """A modified RuntimeInfoHook that safely handles non-scalar tensor values."""

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ``log_vars`` in model outputs every iteration.
        
        This version safely handles multi-element tensor values by taking their mean.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if outputs is not None:
            for key, value in outputs.items():
                if key.startswith('vis_'):
                    continue
                
                # Only process values that we can safely convert to scalars
                if _is_tensor_safe_for_scalar(value):
                    try:
                        # Convert to scalar safely
                        scalar_value = _make_scalar_safe(value)
                        # Use update_info instead of update_scalar to avoid validation
                        runner.message_hub.update_info(f'train/{key}', scalar_value)
                    except Exception as e:
                        # If conversion fails, skip this value
                        print(f"Warning: Could not convert {key} to scalar: {e}")
                        continue 