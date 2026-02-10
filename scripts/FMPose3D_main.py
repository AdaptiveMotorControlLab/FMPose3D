"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import datetime
import logging
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from fmpose3d.common import opts, Human36mDataset, Fusion
from fmpose3d.common.utils import *

from fmpose3d.aggregation_methods import aggregation_RPEA_joint_level

args = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Support loading the model class from a specific file path if provided
CFM = None

if getattr(args, "model_path", ""):
    # Load model from local file path (for custom models)
    import importlib.util
    import pathlib

    model_abspath = os.path.abspath(args.model_path)
    module_name = pathlib.Path(model_abspath).stem
    spec = importlib.util.spec_from_file_location(module_name, model_abspath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    CFM = getattr(module, "Model")
else:
    # Load model from registered model registry
    from fmpose3d.models import get_model
    CFM = get_model(args.model_type)


def test_multi_hypothesis(
    args,
    actions,
    dataLoader,
    model,
    optimizer=None,
    epoch=None,
    hypothesis_num=None,
    steps=None,
):

    model_3d = model["CFM"]
    model_3d.eval()
    split = "test"

    # determine which steps to evaluate (extracted from function; can be provided by caller)
    if steps is None:
        eval_steps = sorted(
            {
                int(s)
                for s in getattr(args, "eval_sample_steps", "3").split(",")
                if str(s).strip()
            }
        )
    else:
        if isinstance(steps, (list, tuple, set)):
            eval_steps = sorted({int(s) for s in steps})
        else:
            eval_steps = [int(steps)]
    action_error_sum_multi = {s: define_error_list(actions) for s in eval_steps}

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable(
            split, [input_2D, gt_3D, batch_cam, scale, bb_box]
        )

        # When test_augmentation=True, input_2D has an extra aug dimension: (B,2,F,J,2)
        # When test_augmentation=False, input_2D has shape: (B,F,J,2)
        if args.test_augmentation:
            input_2D_nonflip = input_2D[:, 0]
            input_2D_flip = input_2D[:, 1]
        else:
            input_2D_nonflip = input_2D
            input_2D_flip = None
        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        # Simple Euler sampler for CFM at test time
        def euler_sample(x2d, y_local, steps):
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full(
                    (gt_3D.size(0), 1, 1, 1),
                    s * dt,
                    device=gt_3D.device,
                    dtype=gt_3D.dtype,
                )
                v_s = model_3d(x2d, y_local, t_s)
                y_local = y_local + dt * v_s
            return y_local

        for s_keep in eval_steps:
            list_hypothesis = []
            for i in range(hypothesis_num):

                y = torch.randn_like(gt_3D)
                y_s = euler_sample(input_2D_nonflip, y, s_keep)

                if args.test_augmentation:
                    y_flip = torch.randn_like(gt_3D)
                    y_flip_s = euler_sample(input_2D_flip, y_flip, s_keep)
                    y_flip_s[:, :, :, 0] *= -1
                    y_flip_s[:, :, args.joints_left + args.joints_right, :] = y_flip_s[
                        :, :, args.joints_right + args.joints_left, :
                    ]
                    y_flip_s = y_flip_s[:, args.pad].unsqueeze(1)
                    y_flip_s[:, :, 0, :] = 0
                    list_hypothesis.append(y_flip_s)

                # per-step metrics only; do not store per-sample outputs
                output_3D_s = y_s[:, args.pad].unsqueeze(1)
                output_3D_s[:, :, 0, :] = 0
                list_hypothesis.append(output_3D_s)

            output_3D_s = aggregation_RPEA_joint_level(args,
                list_hypothesis, batch_cam, input_2D_nonflip, gt_3D, args.topk
            )

            # accumulate by action across the entire test set
            action_error_sum_multi[s_keep] = test_calculation(
                output_3D_s,
                out_target,
                action,
                action_error_sum_multi[s_keep],
                args.dataset,
                subject,
            )

    # aggregate default metrics
    per_step_p1 = {}
    per_step_p2 = {}
    for s_keep in sorted(action_error_sum_multi.keys()):
        p1_s, p2_s = print_error(
            args.dataset, action_error_sum_multi[s_keep], args.train
        )
        per_step_p1[s_keep] = float(p1_s)
        per_step_p2[s_keep] = float(p2_s)

    return per_step_p1, per_step_p2


def train(opt, train_loader, model, optimizer):
    loss_all = {"loss": AccumLoss()}
    model_3d = model["CFM"]
    model_3d.train()
    split = "train"

    for i, data in enumerate(tqdm(train_loader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable(
            split, [input_2D, gt_3D, batch_cam, scale, bb_box]
        )

        if split == "train":
            B, F, J, C = input_2D.shape

            x0_noise = torch.randn(B, F, J, 3, device=gt_3D.device, dtype=gt_3D.dtype)
            x0 = x0_noise

            # t on correct device/dtype and broadcastable: (B,1,1,1)
            t = torch.rand(B, 1, 1, 1, device=gt_3D.device, dtype=gt_3D.dtype)
            y_t = (1.0 - t) * x0 + t * gt_3D
            v_target = gt_3D - x0
            v_pred = model_3d(input_2D, y_t, t)

            loss = ((v_pred - v_target) ** 2).mean()
            N = input_2D.size(0)
            loss_all["loss"].update(loss.detach().cpu().numpy() * N, N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_all["loss"].avg


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {"p1": 0.0, "p2": 0.0}
    mean_error_all = {"p1": AccumLoss(), "p2": AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))
        logging.info("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():

        mean_error_each["p1"] = action_error_sum[action]["p1"].avg * 1000.0
        mean_error_all["p1"].update(mean_error_each["p1"], 1)

        mean_error_each["p2"] = action_error_sum[action]["p2"].avg * 1000.0
        mean_error_all["p2"].update(mean_error_each["p2"], 1)

        if is_train == 0:
            print(
                "{0:<12} {1:>6.2f} {2:>10.2f}".format(
                    action, mean_error_each["p1"], mean_error_each["p2"]
                )
            )
            logging.info(
                "{0:<12} {1:>6.2f} {2:>10.2f}".format(
                    action, mean_error_each["p1"], mean_error_each["p2"]
                )
            )

    if is_train == 0:
        print(
            "{0:<12} {1:>6.4f} {2:>10.4f}".format(
                "Average", mean_error_all["p1"].avg, mean_error_all["p2"].avg
            )
        )
        logging.info(
            "{0:<12} {1:>6.4f} {2:>10.4f}".format(
                "Average", mean_error_all["p1"].avg, mean_error_all["p2"].avg
            )
        )

    return mean_error_all["p1"].avg, mean_error_all["p2"].avg


if __name__ == "__main__":
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # allow overriding timestamp folder by user-provided folder_name
    logtime = time.strftime("%y%m%d_%H%M_%S")
    args.create_time = logtime

    if args.folder_name != "":
        folder_name = args.folder_name
    else:
        folder_name = logtime

    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = "./debug/" + folder_name
        elif args.train:
            args.checkpoint = "./checkpoint/" + folder_name
        elif args.train == False:
            # create a new folder for the test results
            args.previous_dir = os.path.dirname(args.model_weights_path)
            args.checkpoint = os.path.join(args.previous_dir, folder_name)

        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        # backup files
        import shutil
        script_path = os.path.abspath(__file__)
        script_name = os.path.basename(script_path)
        shutil.copyfile(
            src=script_path,
            dst=os.path.join(args.checkpoint, args.create_time + "_" + script_name),
        )
        sh_base = os.path.basename(args.sh_file)
        dst_name = f"{args.create_time}_" + sh_base
        shutil.copyfile(src=args.sh_file, dst=os.path.join(args.checkpoint, dst_name))

        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=os.path.join(args.checkpoint, "train.log"),
            level=logging.INFO,
        )

        arguments = dict(
            (name, getattr(args, name))
            for name in dir(args)
            if not name.startswith("_")
        )
        file_name = os.path.join(args.checkpoint, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> Args:\n")
            for k, v in sorted(arguments.items()):
                opt_file.write("  %s: %s\n" % (str(k), str(v)))
            opt_file.write("==> Args:\n")

    root_path = args.root_path
    dataset_path = root_path + "data_3d_" + args.dataset + ".npz"

    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    if args.train:
        train_data = Fusion(opt=args, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True,
        )
    if args.test:
        test_data = Fusion(opt=args, train=False, dataset=dataset, root_path=root_path)
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = {}
    model["CFM"] = CFM(args).to(device)

    if args.reload:
        model_dict = model["CFM"].state_dict()
        model_path = args.model_weights_path
        print(model_path)
        pre_dict = torch.load(model_path, map_location=device, weights_only=True)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model["CFM"].load_state_dict(model_dict)
        print("Load model Successfully!")

    all_param = []
    all_paramters = 0
    lr = args.lr
    all_param += list(model["CFM"].parameters())
    print(all_paramters)
    logging.info(all_paramters)
    optimizer = optim.Adam(all_param, lr=args.lr, amsgrad=True)
    starttime = datetime.datetime.now()
    best_epoch = 0

    for epoch in range(1, args.nepoch):

        if args.train:
            loss = train(args, train_dataloader, model, optimizer)

        # parse hypotheses list and eval steps
        hypothesis_list = [int(x) for x in args.num_hypothesis_list.split(",")]
        eval_steps_list = [
            int(s)
            for s in str(getattr(args, "eval_sample_steps", "3")).split(",")
            if str(s).strip()
        ]

        # track global best across all (step, hypothesis) for training save logic
        best_global_p1 = None
        best_global_p2 = None
        best_global_pair = None  # (step, hypothesis)

        for s_eval in eval_steps_list:
            p1_by_hyp = {}
            p2_by_hyp = {}
            for hypothesis_num in hypothesis_list:
                print(f"Evaluating step {s_eval} with {hypothesis_num} hypotheses")
                logging.info(
                    f"Evaluating step {s_eval} with {hypothesis_num} hypotheses"
                )
                with torch.no_grad():
                    p1_per_step, p2_per_step = test_multi_hypothesis(
                        args,
                        actions,
                        test_dataloader,
                        model,
                        hypothesis_num=hypothesis_num,
                        steps=s_eval,
                    )

                p1 = p1_per_step[int(s_eval)]
                p2 = p2_per_step[int(s_eval)]
                p1_by_hyp[int(hypothesis_num)] = float(p1)
                p2_by_hyp[int(hypothesis_num)] = float(p2)

                if best_global_p1 is None or float(p1) < best_global_p1:
                    best_global_p1 = float(p1)
                    best_global_p2 = float(p2)
                    best_global_pair = (int(s_eval), int(hypothesis_num))

            # print one line per step with all hypotheses results
            hyp_sorted = sorted(p1_by_hyp.keys())
            hyp_strs = [
                f"h{h}_p1: {p1_by_hyp[h]:.4f}, h{h}_p2: {p2_by_hyp[h]:.4f}"
                for h in hyp_sorted
            ]
            print("step: %d | %s" % (s_eval, " | ".join(hyp_strs)))
            logging.info("step: %d | %s" % (s_eval, " | ".join(hyp_strs)))

        # training summary and checkpointing using best across all (step, hypothesis)
        if args.train and best_global_p1 is not None:
            data_threshold = best_global_p1
            saved_path = save_top_N_models(
                args.previous_name,
                args.checkpoint,
                epoch,
                data_threshold,
                model["CFM"],
                "CFM",
                num_saved_models=getattr(args, "num_saved_models", 3),
            )
            if data_threshold < args.previous_best_threshold:
                args.previous_best_threshold = data_threshold
                args.previous_name = saved_path
                best_epoch = epoch
            print(
                "e: %d, lr: %.7f, loss: %.4f, best_p1: %.4f, best_p2: %.4f, best_pair: step %d, hyp %d"
                % (
                    epoch,
                    lr,
                    loss,
                    best_global_p1,
                    best_global_p2,
                    best_global_pair[0],
                    best_global_pair[1],
                )
            )
            logging.info(
                "epoch: %d, lr: %.7f, loss: %.4f, best_p1: %.4f, best_p2: %.4f, best_pair: step %d, hyp %d"
                % (
                    epoch,
                    lr,
                    loss,
                    best_global_p1,
                    best_global_p2,
                    best_global_pair[0],
                    best_global_pair[1],
                )
            )
        elif not args.train:
            break

        if epoch % args.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.lr_decay_large
                lr *= args.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.lr_decay
                lr *= args.lr_decay

    endtime = datetime.datetime.now()
    a = (endtime - starttime).seconds
    h = a // 3600
    mins = (a - 3600 * h) // 60
    s = a - 3600 * h - mins * 60

    print(
        "best epoch:{}, best result(mpjpe):{}".format(
            best_epoch, args.previous_best_threshold
        )
    )
    logging.info(
        "best epoch:{}, best result(mpjpe):{}".format(
            best_epoch, args.previous_best_threshold
        )
    )
    print(h, "h", mins, "mins", s, "s")
    logging.info("training time: %dh,%dmin%ds" % (h, mins, s))
    print(args.checkpoint)
    logging.info(args.checkpoint)
