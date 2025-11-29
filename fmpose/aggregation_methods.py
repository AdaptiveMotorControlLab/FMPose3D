import torch
from fmpose.common.utils import project_to_2d

def average_aggregation(list_hypothesis):
    return torch.mean(torch.stack(list_hypothesis), dim=0)


def aggregation_select_single_best_hypothesis_by_2D_error(args,
    list_hypothesis, batch_cam, input_2D, gt_3D
):
    """
    Select per-joint 3D from the hypothesis whose 2D projection yields minimal L2 error.

    Args:
        list_hypothesis: list of (B,1,J,3) tensors
        batch_cam: (B, 9) or (B, 1, 9) intrinsics [f(2), c(2), k(3), p(2)]
        input_2D: (B, F, J, 2) 2D joints in image coordinates
        gt_3D: (B, F, J, 3) used for shape metadata only
    Returns:
        (B,1,J,3) aggregated 3D pose with joint 0 set to 0
    """
    if len(list_hypothesis) == 0:
        raise ValueError("list_hypothesis is empty")

    device = list_hypothesis[0].device
    dtype = list_hypothesis[0].dtype

    # Shapes
    B = gt_3D.size(0)
    J = gt_3D.size(2)
    F = gt_3D.size(1)
    assert F >= 1, "Expected at least one frame"

    # Stack hypotheses: (H,B,1,J,3) -> (B,H,J,3)
    stack = torch.stack(list_hypothesis, dim=0)  # (H,B,1,J,3)
    X_hbj3 = stack[:, :, 0, :, :]  # (H,B,J,3)
    X_bhj3 = X_hbj3.transpose(0, 1).contiguous()  # (B,H,J,3)
    H = X_bhj3.size(1)

    # Prepare camera params: (B,9)
    if batch_cam.dim() == 3 and batch_cam.size(1) == 1:
        cam_b9 = batch_cam[:, 0, :].contiguous()
    elif batch_cam.dim() == 2 and batch_cam.size(1) == 9:
        cam_b9 = batch_cam
    else:
        cam_b9 = batch_cam.view(B, -1)
    assert cam_b9.size(1) == 9, f"camera params should be 9-dim, got {cam_b9.size()}"

    # Target 2D at the same frame index as 3D selection (args.pad)
    # input_2D: (B,F,J,2) -> (B,J,2)
    target_2d = input_2D[:, getattr(args, "pad", 0)].contiguous()  # (B,J,2)

    # Convert hypotheses from root-relative to absolute camera coordinates using GT root
    # Root at frame args.pad: (B,3)
    gt_root = gt_3D[:, getattr(args, "pad", 0), 0, :].contiguous()  # (B,3)
    X_abs = X_bhj3.clone()
    X_abs[:, :, 1:, :] = X_abs[:, :, 1:, :] + gt_root.unsqueeze(1).unsqueeze(1)
    X_abs[:, :, 0, :] = gt_root.unsqueeze(1)

    # Vectorized projection for all hypotheses in absolute coordinates
    # (B,H,J,3) -> (B*H,J,3)
    X_flat = X_abs.view(B * H, J, 3)
    cam_rep = cam_b9.repeat_interleave(H, dim=0)  # (B*H,9)

    # project_to_2d expects last dim=3 and cam (N,9)
    # Returns normalized coordinates (when crop_uv=0) because camera params are normalized
    proj2d_flat = project_to_2d(X_flat, cam_rep)  # (B*H,J,2) normalized coordinates
    proj2d_bhj = proj2d_flat.view(B, H, J, 2)

    # Per-hypothesis per-joint 2D error (both in normalized coordinates)
    diff = proj2d_bhj - target_2d.unsqueeze(1)  # (B,H,J,2)
    dist = torch.norm(diff, dim=-1)  # (B,H,J)

    # Exclude root joint (0) due to undefined depth when using root-relative 3D
    dist[:, :, 0] = float("inf")

    # Argmin across hypotheses per joint
    best_h = torch.argmin(dist, dim=1)  # (B,J)

    # Gather 3D using advanced indexing (return root-relative coordinates)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, J)  # (B,J)
    j_idx = torch.arange(J, device=device).unsqueeze(0).expand(B, J)  # (B,J)
    selected_bj3 = X_bhj3[b_idx, best_h, j_idx, :]  # (B,J,3)

    agg = selected_bj3.unsqueeze(1).to(dtype=dtype)
    agg[:, :, 0, :] = 0
    return agg


def aggregation_RPEA_weighted_by_2D_error(
    args, list_hypothesis, batch_cam, input_2D, gt_3D, topk=3
):
    """
    Select per-joint 3D from the hypothesis whose 2D projection yields minimal L2 error.

    Args:
        list_hypothesis: list of (B,1,J,3) tensors
        batch_cam: (B, 9) or (B, 1, 9) intrinsics [f(2), c(2), k(3), p(2)]
        input_2D: (B, F, J, 2) 2D joints in image coordinates
        gt_3D: (B, F, J, 3) used for shape metadata only
    Returns:
        (B,1,J,3) aggregated 3D pose with joint 0 set to 0
    """
    if len(list_hypothesis) == 0:
        raise ValueError("list_hypothesis is empty")

    device = list_hypothesis[0].device
    dtype = list_hypothesis[0].dtype

    # Shapes
    B = gt_3D.size(0)
    J = gt_3D.size(2)
    F = gt_3D.size(1)
    assert F >= 1, "Expected at least one frame"

    # Stack hypotheses: (H,B,1,J,3) -> (B,H,J,3)
    stack = torch.stack(list_hypothesis, dim=0)  # (H,B,1,J,3)
    X_hbj3 = stack[:, :, 0, :, :]  # (H,B,J,3)
    X_bhj3 = X_hbj3.transpose(0, 1).contiguous()  # (B,H,J,3)
    H = X_bhj3.size(1)

    # Prepare camera params: (B,9)
    if batch_cam.dim() == 3 and batch_cam.size(1) == 1:
        cam_b9 = batch_cam[:, 0, :].contiguous()
    elif batch_cam.dim() == 2 and batch_cam.size(1) == 9:
        cam_b9 = batch_cam
    else:
        cam_b9 = batch_cam.view(B, -1)
    assert cam_b9.size(1) == 9, f"camera params should be 9-dim, got {cam_b9.size()}"

    # Target 2D at the same frame index as 3D selection (args.pad)
    # input_2D: (B,F,J,2) -> (B,J,2)
    target_2d = input_2D[:, getattr(args, "pad", 0)].contiguous()  # (B,J,2)

    # Convert hypotheses from root-relative to absolute camera coordinates using GT root
    # Root at frame args.pad: (B,3)
    gt_root = gt_3D[:, getattr(args, "pad", 0), 0, :].contiguous()  # (B,3)
    X_abs = X_bhj3.clone()
    X_abs[:, :, 1:, :] = X_abs[:, :, 1:, :] + gt_root.unsqueeze(1).unsqueeze(1)
    X_abs[:, :, 0, :] = gt_root.unsqueeze(1)

    # Vectorized projection for all hypotheses in absolute coordinates
    # (B,H,J,3) -> (B*H,J,3)
    X_flat = X_abs.view(B * H, J, 3)
    cam_rep = cam_b9.repeat_interleave(H, dim=0)  # (B*H,9)

    # project_to_2d expects last dim=3 and cam (N,9)
    proj2d_flat = project_to_2d(X_flat, cam_rep)  # (B*H,J,2)
    proj2d_bhj = proj2d_flat.view(B, H, J, 2)

    # Per-hypothesis per-joint 2D error
    diff = proj2d_bhj - target_2d.unsqueeze(1)  # (B,H,J,2)
    dist = torch.norm(diff, dim=-1)  # (B,H,J)

    # For root joint (0), avoid NaNs in softmax by setting equal distances
    # This yields uniform weights at the root (we set root to 0 later anyway)
    dist[:, :, 0] = 0.0

    # Convert 2D losses to weights using softmax over top-k hypotheses per joint
    tau = float(getattr(args, "weight_softmax_tau", 1.0))
    H = dist.size(1)
    k = int(getattr(args, "topk", None))
    # print("k:", k)
    # k = int(H//2)+1
    k = max(1, min(k, H))

    # top-k smallest distances along hypothesis dim
    topk_vals, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)  # (B,k,J)

    # Weight calculation method ; weight_method = 'exp'
    temp = args.exp_temp
    max_safe_val = temp * 20
    topk_vals_clipped = torch.clamp(topk_vals, max=max_safe_val)
    exp_vals = torch.exp(-topk_vals_clipped / temp)
    exp_sum = exp_vals.sum(dim=1, keepdim=True)
    topk_weights = exp_vals / torch.clamp(exp_sum, min=1e-10)
    nan_mask = torch.isnan(topk_weights).any(dim=1, keepdim=True)
    uniform_weights = torch.ones_like(topk_weights) / k
    topk_weights = torch.where(
        nan_mask.expand_as(topk_weights), uniform_weights, topk_weights
    )

    # scatter back to full H with zeros elsewhere
    weights = torch.zeros_like(dist)  # (B,H,J)
    weights.scatter_(1, topk_idx, topk_weights)

    # Weighted sum of root-relative 3D hypotheses per joint
    weights_exp = weights.unsqueeze(-1)  # (B,H,J,1)
    weighted_bj3 = torch.sum(X_bhj3 * weights_exp, dim=1)  # (B,J,3)

    # Assemble output (B,1,J,3) and enforce root at origin
    agg = weighted_bj3.unsqueeze(1).to(dtype=dtype)
    agg[:, :, 0, :] = 0
    return agg