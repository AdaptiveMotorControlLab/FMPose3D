"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def define_error_list(actions):
    return {
        action: {
            "p1": AccumLoss(),
            "p2": AccumLoss(),
            "pck": AccumLoss(),
            "auc": AccumLoss(),
        }
        for action in actions
    }


def define_actions_3dhp(action="*", train=False):
    if action != "*":
        raise ValueError("MPI-INF-3DHP test annotations do not include action labels; use --actions '*'.")
    return ["Seq1", "Seq2"] if train else ["Seq1"]
