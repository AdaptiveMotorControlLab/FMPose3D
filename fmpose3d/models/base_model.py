"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}


def register_model(name: str):
    """Class decorator that registers a model under *name*.

    Usage::

        @register_model("my_model")
        class MyModel(BaseModel):
            ...

    The model can then be retrieved with :func:`get_model`.
    """

    def decorator(cls: type["BaseModel"]) -> type["BaseModel"]:
        if name in _MODEL_REGISTRY:
            warnings.warn(
                f"Model '{name}' is already registered "
                f"(existing: {_MODEL_REGISTRY[name].__qualname__}, "
                f"new: {cls.__qualname__})"
            )
            # raise ValueError(
            #     f"Model '{name}' is already registered "
            #     f"(existing: {_MODEL_REGISTRY[name].__qualname__}, "
            #     f"new: {cls.__qualname__})"
            # )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type["BaseModel"]:
    """Return the model class registered under *name*.

    Raises :class:`KeyError` with a helpful message when the name is unknown.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class BaseModel(ABC, nn.Module):
    """Abstract base class for all FMPose3D lifting models.

    Every model must accept a single *args* namespace / object in its
    constructor and implement :meth:`forward` with the signature below.

    Parameters expected on *args* (at minimum):
        - ``channel``   – embedding dimension
        - ``layers``    – number of transformer / GCN blocks
        - ``d_hid``     – hidden MLP dimension
        - ``token_dim`` – token dimension
        - ``n_joints``  – number of body joints
    """

    @abstractmethod
    def __init__(self, args) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        pose_2d: torch.Tensor,
        y_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field for flow matching.

        Args:
            pose_2d: 2D keypoints, shape ``(B, F, J, 2)``.
            y_t: Noisy 3D hypothesis at time *t*, shape ``(B, F, J, 3)``.
            t: Diffusion / flow time, shape ``(B, F, 1, 1)`` with values
               in ``[0, 1]``.

        Returns:
            Predicted velocity ``v``, shape ``(B, F, J, 3)``.
        """
        ...

