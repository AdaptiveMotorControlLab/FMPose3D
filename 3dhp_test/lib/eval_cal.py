"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import numpy as np
import torch


def mpjpe(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))


def pck(predicted, target):
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape) - 1)
    threshold = torch.tensor(0.150, dtype=dis.dtype, device=dis.device)
    return (dis < threshold).float().mean()


def auc(predicted, target):
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape) - 1)
    thresholds = torch.arange(0, 151, 5, dtype=dis.dtype, device=dis.device) / 1000.0
    threshold_shape = (-1,) + (1,) * dis.ndim
    return (dis.unsqueeze(0) < thresholds.view(threshold_shape)).float().mean()


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    if data_type == "h36m" or data_type.startswith("3dhp"):
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    if data_type.startswith("3dhp"):
        error_sum = mpjpe_by_action_pck(predicted, target, action, error_sum)
        error_sum = mpjpe_by_action_auc(predicted, target, action, error_sum)
    return error_sum


def _action_name(action):
    end_index = action.find(" ")
    if end_index != -1:
        return action[:end_index]
    return action


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        action_name = _action_name(action[0])
        action_error_sum[action_name]["p1"].update(torch.mean(dist).item() * num, num)
    else:
        for i in range(num):
            action_name = _action_name(action[i])
            action_error_sum[action_name]["p1"].update(dist[i].item(), 1)
    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        action_name = _action_name(action[0])
        action_error_sum[action_name]["p2"].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            action_name = _action_name(action[i])
            action_error_sum[action_name]["p2"].update(np.mean(dist), 1)
    return action_error_sum


def mpjpe_by_action_pck(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)

    if len(set(list(action))) == 1:
        action_name = _action_name(action[0])
        action_error_sum[action_name]["pck"].update(pck(predicted, target).item() * num, num)
    else:
        for i in range(num):
            action_name = _action_name(action[i])
            action_error_sum[action_name]["pck"].update(pck(predicted[i : i + 1], target[i : i + 1]).item(), 1)
    return action_error_sum


def mpjpe_by_action_auc(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)

    if len(set(list(action))) == 1:
        action_name = _action_name(action[0])
        action_error_sum[action_name]["auc"].update(auc(predicted, target).item() * num, num)
    else:
        for i in range(num):
            action_name = _action_name(action[i])
            action_error_sum[action_name]["auc"].update(auc(predicted[i : i + 1], target[i : i + 1]).item(), 1)
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)
    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))
    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / norm_y
    t = mu_x - a * np.matmul(mu_y, r)
    predicted_aligned = a * np.matmul(predicted, r) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)
