"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fmpose3d.aggregation_methods import aggregation_RPEA_joint_level
from fmpose3d.models import get_model

from lib.camera import camera_params_for_subject
from lib.dataset_3dhp import ThreeDHPTestDataset
from lib.utils import AccumLoss, define_actions_3dhp, define_error_list
import lib.eval_cal as eval_cal


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean 3DHP inference with processed test npz.")
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "dataset" / "data_test_3dhp.npz")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a Model definition. Defaults to the package FMPose3D human model.",
    )
    parser.add_argument("--model-type", default="fmpose3d_humans", type=str)
    parser.add_argument("--saved-model-path", type=Path, default=ROOT / "pretrained" / "fmpose3d_h36m" / "FMpose3D_pretrained_weights.pth")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--folder-name", type=str, default="")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--frames", default=1, type=int)
    parser.add_argument("--layers", default=5, type=int)
    parser.add_argument("--channel", default=512, type=int)
    parser.add_argument("--d-hid", default=1024, type=int)
    parser.add_argument("--token-dim", default=256, type=int)
    parser.add_argument("--n-joints", default=17, type=int)
    parser.add_argument("--dataset", default="3dhp_valid", type=str)
    parser.add_argument("--actions", default="*", type=str)
    parser.add_argument("--subjects-test", default="TS1,TS2,TS3,TS4,TS5,TS6", type=str)
    parser.add_argument("--eval-sample-steps", default="2", type=str)
    parser.add_argument("--num-hypothesis-list", default="1", type=str)
    parser.add_argument("--topk", default=6, type=int)
    parser.add_argument("--exp-temp", default=0.005, type=float)
    parser.add_argument("--test-augmentation", default=True, type=str2bool)
    parser.add_argument("--test-augmentation-flip-hypothesis", default=True, type=str2bool)
    parser.add_argument("--max-batches", default=0, type=int, help="Smoke test limit. 0 means full evaluation.")
    parser.add_argument("--manual-seed", default=1, type=int)
    args = parser.parse_args()

    args.pad = (args.frames - 1) // 2
    args.root_joint = 0
    args.train = 0
    args.test = True
    args.keypoints = "gt_17_univ"
    args.joints_left = [4, 5, 6, 11, 12, 13]
    args.joints_right = [1, 2, 3, 14, 15, 16]
    args.kps_left = args.joints_left
    args.kps_right = args.joints_right
    return args


def configure_reproducibility(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_class(model_path, model_type):
    if model_path is None:
        return get_model(model_type)
    model_path = Path(model_path).resolve()
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, "Model")


def get_device(gpu):
    if gpu in {"", "-1", "cpu", "none", "None"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(f"CUDA is not available; running on CPU instead of GPU {gpu}.")
    return torch.device("cpu")


def camera_tensor_for_subjects(subjects, device, dtype=torch.float32):
    cam_params = [camera_params_for_subject(subject) for subject in subjects]
    return torch.tensor(cam_params, dtype=dtype, device=device)


def print_error(data_type, action_error_sum, is_train):
    if data_type == "h36m" or data_type.startswith("3dhp"):
        return print_error_action(action_error_sum, is_train, data_type)
    return 0, 0, 0, 0


def print_error_action(action_error_sum, is_train, data_type):
    mean_error_each = {"p1": 0.0, "p2": 0.0, "pck": 0.0, "auc": 0.0}
    mean_error_all = {"p1": AccumLoss(), "p2": AccumLoss(), "pck": AccumLoss(), "auc": AccumLoss()}

    if not is_train:
        if data_type.startswith("3dhp"):
            print("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
            logging.info("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
        else:
            print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action in action_error_sum.keys():
        if not is_train:
            print("{0:<12} ".format(action), end="")

        mean_error_each["p1"] = action_error_sum[action]["p1"].avg * 1000.0
        mean_error_all["p1"].update(mean_error_each["p1"], 1)
        mean_error_each["p2"] = action_error_sum[action]["p2"].avg * 1000.0
        mean_error_all["p2"].update(mean_error_each["p2"], 1)
        mean_error_each["pck"] = action_error_sum[action]["pck"].avg * 100.0
        mean_error_all["pck"].update(mean_error_each["pck"], 1)
        mean_error_each["auc"] = action_error_sum[action]["auc"].avg * 100.0
        mean_error_all["auc"].update(mean_error_each["auc"], 1)

        if is_train == 0:
            if data_type.startswith("3dhp"):
                print(
                    "{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                        mean_error_each["p1"],
                        mean_error_each["p2"],
                        mean_error_each["pck"],
                        mean_error_each["auc"],
                    )
                )
                logging.info(
                    "{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                        action,
                        mean_error_each["p1"],
                        mean_error_each["p2"],
                        mean_error_each["pck"],
                        mean_error_each["auc"],
                    )
                )
            else:
                print("{0:>6.2f} {1:>10.2f}".format(mean_error_each["p1"], mean_error_each["p2"]))

    if is_train == 0:
        if data_type.startswith("3dhp"):
            print(
                "{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                    "Average",
                    mean_error_all["p1"].avg,
                    mean_error_all["p2"].avg,
                    mean_error_all["pck"].avg,
                    mean_error_all["auc"].avg,
                )
            )
            logging.info(
                "{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                    "Average",
                    mean_error_all["p1"].avg,
                    mean_error_all["p2"].avg,
                    mean_error_all["pck"].avg,
                    mean_error_all["auc"].avg,
                )
            )
        else:
            print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all["p1"].avg, mean_error_all["p2"].avg))

    if data_type.startswith("3dhp"):
        return mean_error_all["p1"].avg, mean_error_all["p2"].avg, mean_error_all["pck"].avg, mean_error_all["auc"].avg
    return mean_error_all["p1"].avg, mean_error_all["p2"].avg, 0, 0


def test(actions, dataloader, model, args, hypothesis_num=1):
    model.eval()
    eval_steps = sorted({int(s) for s in str(args.eval_sample_steps).split(",") if str(s).strip()})
    action_error_sum_multi = {s: define_error_list(actions) for s in eval_steps}

    print(f"\n{'=' * 80}")
    print(f"Testing with {hypothesis_num} hypothesis(es), eval_steps: {eval_steps}")
    print(f"{'=' * 80}\n")

    for i, data in enumerate(tqdm(dataloader, 0)):
        _, gt_3d, input_2d, _, action, subject, _ = data
        input_2d = input_2d.contiguous().to(args.device, dtype=torch.float32)
        gt_3d = gt_3d.contiguous().to(args.device, dtype=torch.float32)

        input_2d_nonflip = input_2d[:, 0]
        input_2d_flip = input_2d[:, 1] if input_2d.size(1) > 1 else input_2d[:, 0]

        out_target = gt_3d.clone()
        out_target[:, :, args.root_joint] = 0

        def euler_sample(x2d, y_local, steps, model_3d):
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full((gt_3d.size(0), 1, 1, 1), s * dt, device=gt_3d.device, dtype=gt_3d.dtype)
                v_s = model_3d(x2d, y_local, t_s)
                y_local = y_local + dt * v_s
            return y_local

        if i == 0:
            print(f"eval_steps: {eval_steps}, hypothesis_num: {hypothesis_num}")

        for s_keep in eval_steps:
            list_hypothesis = []
            for _ in range(hypothesis_num):
                y = torch.randn_like(gt_3d)
                y_s = euler_sample(input_2d_nonflip, y, s_keep, model)

                if args.test_augmentation_flip_hypothesis:
                    y_flip = torch.randn_like(gt_3d)
                    y_flip[:, :, :, 0] *= -1
                    y_flip[:, :, args.joints_left + args.joints_right, :] = y_flip[
                        :, :, args.joints_right + args.joints_left, :
                    ]
                    y_flip_s = euler_sample(input_2d_flip, y_flip, s_keep, model)
                    y_flip_s[:, :, :, 0] *= -1
                    y_flip_s[:, :, args.joints_left + args.joints_right, :] = y_flip_s[
                        :, :, args.joints_right + args.joints_left, :
                    ]
                    y_flip_s_frame = y_flip_s[:, args.pad].unsqueeze(1)
                    y_flip_s_frame[:, :, 0, :] = 0
                    list_hypothesis.append(y_flip_s_frame)

                y_s_frame = y_s[:, args.pad].unsqueeze(1)
                y_s_frame[:, :, 0, :] = 0
                list_hypothesis.append(y_s_frame)

            cam_tensor = camera_tensor_for_subjects(subject, gt_3d.device, dtype=gt_3d.dtype)
            output_3d_s = aggregation_RPEA_joint_level(
                args, list_hypothesis, cam_tensor, input_2d_nonflip, gt_3d
            )
            action_error_sum_multi[s_keep] = eval_cal.test_calculation(
                output_3d_s, out_target, action, action_error_sum_multi[s_keep], args.dataset, subject
            )

        if args.max_batches and i + 1 >= args.max_batches:
            break

    per_step_p1 = {}
    per_step_p2 = {}
    per_step_pck = {}
    per_step_auc = {}
    for s_keep in sorted(action_error_sum_multi.keys()):
        p1_s, p2_s, pck_s, auc_s = print_error(args.dataset, action_error_sum_multi[s_keep], args.train)
        per_step_p1[s_keep] = float(p1_s)
        per_step_p2[s_keep] = float(p2_s)
        per_step_pck[s_keep] = float(pck_s)
        per_step_auc[s_keep] = float(auc_s)

    return per_step_p1, per_step_p2, per_step_pck, per_step_auc


def setup_logging(args):
    if args.folder_name:
        folder_name = args.folder_name
    else:
        folder_name = (
            f"s_{args.eval_sample_steps}_Top{args.topk}_exp_temp{args.exp_temp}_"
            f"S{args.subjects_test}_h{args.num_hypothesis_list}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    result_dir = args.results_dir / folder_name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "train.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
    return result_dir, log_path


def main():
    args = parse_args()
    if args.gpu not in {"", "-1", "cpu", "none", "None"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = get_device(args.gpu)
    configure_reproducibility(args.manual_seed)

    result_dir, log_path = setup_logging(args)
    print(f"Results: {result_dir}")
    print(f"Log: {log_path}")

    subjects = [s for s in args.subjects_test.split(",") if s]
    dataset = ThreeDHPTestDataset(
        args.dataset_path,
        subjects=subjects,
        test_augmentation=args.test_augmentation,
        kps_left=args.kps_left,
        kps_right=args.kps_right,
        joints_left=args.joints_left,
        joints_right=args.joints_right,
    )
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset summary: {dataset.summary()}")
    logging.info(f"Dataset: {args.dataset_path}")
    logging.info(f"Dataset summary: {dataset.summary()}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=args.device.type == "cuda",
    )

    model_cls = load_model_class(args.model_path, args.model_type)
    model = model_cls(args).to(args.device)

    print(args.saved_model_path)
    pre_dict = torch.load(args.saved_model_path, map_location=args.device, weights_only=True)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print("model loaded successfully!")

    actions = define_actions_3dhp(args.actions, train=False)
    hypothesis_list = [int(x) for x in str(args.num_hypothesis_list).split(",") if str(x).strip()]
    eval_steps_list = [int(s) for s in str(args.eval_sample_steps).split(",") if str(s).strip()]

    best_global_p1 = None
    best_global_p2 = None
    best_global_pck = None
    best_global_auc = None
    best_global_pair = None
    all_metrics = {}

    for s_eval in eval_steps_list:
        p1_by_hyp = {}
        p2_by_hyp = {}
        pck_by_hyp = {}
        auc_by_hyp = {}

        for hypothesis_num in hypothesis_list:
            print(f"\n{'=' * 80}")
            print(f"Evaluating step {s_eval} with {hypothesis_num} hypotheses")
            print(f"{'=' * 80}\n")
            logging.info(f"Evaluating step {s_eval} with {hypothesis_num} hypotheses")

            with torch.no_grad():
                args_backup = args.eval_sample_steps
                args.eval_sample_steps = str(s_eval)
                p1_per_step, p2_per_step, pck_per_step, auc_per_step = test(
                    actions, dataloader, model, args, hypothesis_num=hypothesis_num
                )
                args.eval_sample_steps = args_backup

            p1 = p1_per_step[int(s_eval)]
            p2 = p2_per_step[int(s_eval)]
            pck_s = pck_per_step[int(s_eval)]
            auc_s = auc_per_step[int(s_eval)]

            p1_by_hyp[int(hypothesis_num)] = float(p1)
            p2_by_hyp[int(hypothesis_num)] = float(p2)
            pck_by_hyp[int(hypothesis_num)] = float(pck_s)
            auc_by_hyp[int(hypothesis_num)] = float(auc_s)

            all_metrics[f"step_{s_eval}_hyp_{hypothesis_num}"] = {
                "p1": float(p1),
                "p2": float(p2),
                "pck": float(pck_s),
                "auc": float(auc_s),
            }

            if best_global_p1 is None or float(p1) < best_global_p1:
                best_global_p1 = float(p1)
                best_global_p2 = float(p2)
                best_global_pck = float(pck_s)
                best_global_auc = float(auc_s)
                best_global_pair = (int(s_eval), int(hypothesis_num))

        hyp_sorted = sorted(p1_by_hyp.keys())
        hyp_strs = [
            f"h{h}_p1: {p1_by_hyp[h]:.4f}, h{h}_p2: {p2_by_hyp[h]:.4f}, "
            f"h{h}_pck: {pck_by_hyp[h]:.4f}, h{h}_auc: {auc_by_hyp[h]:.4f}"
            for h in hyp_sorted
        ]
        print("\n" + "=" * 80)
        print(f"Step: {s_eval} | " + " | ".join(hyp_strs))
        print("=" * 80 + "\n")
        logging.info(f"step: {s_eval} | " + " | ".join(hyp_strs))

    if best_global_p1 is not None:
        print("\n" + "=" * 80)
        print(
            f"BEST RESULT: step {best_global_pair[0]}, hyp {best_global_pair[1]}: "
            f"p1: {best_global_p1:.4f}, p2: {best_global_p2:.4f}, "
            f"pck: {best_global_pck:.4f}, auc: {best_global_auc:.4f}"
        )
        print("=" * 80 + "\n")
        logging.info(
            f"BEST: step {best_global_pair[0]}, hyp {best_global_pair[1]}: "
            f"p1: {best_global_p1:.4f}, p2: {best_global_p2:.4f}, "
            f"pck: {best_global_pck:.4f}, auc: {best_global_auc:.4f}"
        )
        all_metrics["best"] = {
            "step": best_global_pair[0],
            "hypothesis": best_global_pair[1],
            "p1": best_global_p1,
            "p2": best_global_p2,
            "pck": best_global_pck,
            "auc": best_global_auc,
        }

    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
