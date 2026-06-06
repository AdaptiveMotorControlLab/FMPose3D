"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

Shared helper for resolving / downloading FMPose3D model weights.
"""

HF_REPO_ID: str = "MLAdaptiveIntelligence/FMPose3D"


def resolve_weights_path(local_path: str, filename: str) -> str:
    """Return a local weights path, downloading from Hugging Face Hub if needed.

    Parameters
    ----------
    local_path : str
        User-supplied local path. If falsy, ``filename`` is fetched from
        the Hugging Face Hub (cached under ``~/.cache/huggingface``).
    filename : str
        The exact remote filename in the FMPose3D Hugging Face repo
        (e.g. ``"fmpose3d_humans.pth"``, ``"fmpose3d_animals.pth"``,
        ``"sa_finetune_hrnet_w32.pt"``).

    Returns
    -------
    str
        Absolute path to the weight file on disk.
    """
    if local_path:
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download model weights. "
            "Install it with:  pip install huggingface_hub\n"
            "Or download the weights manually and pass the local path."
        ) from None

    print(
        f"No local weights path specified. "
        f"Downloading '{filename}' from Hugging Face ({HF_REPO_ID})..."
    )
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
