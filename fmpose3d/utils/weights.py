"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

"""Shared helpers for resolving / downloading FMPose3D model weights."""

HF_REPO_ID: str = "MLAdaptiveIntelligence/FMPose3D"


def resolve_weights_path(model_weights_path: str, model_type: str) -> str:
    """Return a local weights path, downloading from Hugging Face Hub if needed.

    Parameters
    ----------
    model_weights_path : str
        User-supplied local path.  If falsy the weights are fetched from the
        Hugging Face Hub automatically.
    model_type : str
        Model variant name used to derive the remote filename
        (e.g. ``"fmpose3d_humans"`` -> ``fmpose3d_humans.pth``).

    Returns
    -------
    str
        Absolute path to the weight file on disk.
    """
    if model_weights_path:
        return model_weights_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download model weights. "
            "Install it with:  pip install huggingface_hub\n"
            "Or download the weights manually and pass the local path."
        ) from None

    filename = f"{model_type}.pth"
    print(
        f"No local weights path specified. "
        f"Downloading '{filename}' from Hugging Face ({HF_REPO_ID})..."
    )
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
