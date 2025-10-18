import torch
from torch.autograd import Variable
import numpy as np
import os
import time
import hashlib

def gaussian_loss(M, S, target): 
    # M:mu, S:s;  M.shape,s.shape:[B,1,17,3] 
    # target:Ground Truth 3D
    assert M.shape == target.shape
    # return torch.mean((torch.norm(target - M, p=2, dim=-1) * torch.exp(-S) + S))
    S = S.squeeze(-1)
    S = torch.clamp(S, -3)
    return torch.mean(torch.norm(target - M, p=2, dim=-1)*torch.exp(-S) + S)

class AccumLoss(object):
    """
    for initialize and accumulate loss/err
    """
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


def get_varialbe(split, target):
    """
    :param split: 'train' or 'val'
    :param target: a list of tensors
    :return: a list of variables
    """
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)
    return var


def get_uvd2xyz(uvd, gt_3D, cam):
    """
    transfer uvd to xyz

    :param uvd: N*T*V*3 (uv and z channel)
    :param gt_3D: N*T*V*3 (NOTE: V=0 is absolute depth value of root joint)

    :return: root-relative xyz results
    """
    N, T, V,_ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()  # N*T*V*3
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()# N*T*V*3
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # N*T*V*2

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1) # N*T*V*2
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)# N*T*V*2

    # change to global
    z_global = dec_out_all[:, :, :, 2]# N*T*V
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  # N*T*V
    z_global = z_global.unsqueeze(-1)  # N*T*V*1
    
    uv = enc_in_all - cam_c_all  # N*T*V*2
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  # N*T*V*2
    xyz_global = torch.cat((xy, z_global), -1)  # N*T*V*3
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))# N*T*V*3

    return xyz_offset

def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2, pck, auc = 0, 0, 0, 0
    if data_type == 'h36m' or data_type.startswith('3dhp'):
        mean_error_p1, mean_error_p2, pck, auc = print_error_action(action_error_sum, is_train, data_type)
    elif data_type == 'humaneva15':
        mean_error_p1, mean_error_p2, pck, auc = print_error_action_subject(action_error_sum)

    return mean_error_p1, mean_error_p2, pck, auc

def print_error_action_subject(action_error_sum):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}
    subjects_test = ['Validate/S1', 'Validate/S2', 'Validate/S3']

    print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "Subject", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        for subject in subjects_test:
            print("{0:<12} ".format(action), end="")
            print("{0:<12} ".format(subject.split('/')[-1]), end="")

            for j in range(1, 3):
                mean_error_each['p'+str(j)] = action_error_sum[action][subject]['p'+str(j)].avg * 1000.0
                mean_error_all['p'+str(j)].update(mean_error_each['p'+str(j)], 1)

            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg

def print_error_action(action_error_sum, is_train, data_type):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0, 'auc': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}

    if not is_train:
        if data_type.startswith('3dhp'):
            print("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
        else:
            print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if not is_train:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
        mean_error_all['pck'].update(mean_error_each['pck'], 1)

        mean_error_each['auc'] = action_error_sum[action]['auc'].avg * 100.0
        mean_error_all['auc'].update(mean_error_each['auc'], 1)

        if is_train == 0:
            if data_type.startswith('3dhp'):
                print("{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                    mean_error_each['p1'], mean_error_each['p2'], 
                    mean_error_each['pck'], mean_error_each['auc']))
            else:
                print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        if data_type.startswith('3dhp'):
            print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                "Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg,
                mean_error_all['pck'].avg, mean_error_all['auc'].avg))
        else:
            print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))

    if data_type.startswith('3dhp'):
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg,  \
                mean_error_all['pck'].avg, mean_error_all['auc'].avg
    else:
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg, 0, 0

def save_model(previous_name, save_dir, epoch, data_threshold, model, model_name):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    # if multi_gpu:
    #     torch.save(model.module.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    # else:
    #     torch.save(model.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    
    torch.save(model.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 1000))
    
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 1000)
    
    return previous_name


def save_model_multi_gpu(previous_name, save_dir, epoch, data_threshold, model, model_name):
    if os.path.exists(previous_name):
        os.remove(previous_name)
    
    torch.save(model.module.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)
    
    return previous_name
    
def define_error_list(actions):
    """
    define error sum_list
    error_sum: the return list
    actions: action list
    subjects: subjects list, if no subjects only make the list with actions
    :return: {action1:{'p1':, 'p2':},action2:{'p1':, 'p2':}}...
    """
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss(), 'pck':AccumLoss(), 'auc':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum

def define_error_list_subject(actions):
    """
    define error sum_list
    error_sum: the return list
    actions: action list
    subjects: subjects list, if no subjects only make the list with actions
    :return: {action1:{'p1':, 'p2':},action2:{'p1':, 'p2':}}...
    """
    subjects_test = ['Validate/S1', 'Validate/S2', 'Validate/S3']

    error_sum = {}
    error_sum.update({actions[i]: 
        {
            subjects_test[0]: {'p1':AccumLoss(), 'p2':AccumLoss()}, \
            subjects_test[1]: {'p1':AccumLoss(), 'p2':AccumLoss()}, \
            subjects_test[2]: {'p1':AccumLoss(), 'p2':AccumLoss()}
        } 
        for i in range(len(actions))})

    return error_sum

def back_to_ori_uv(cropped_uv,bb_box):
    """
    for cropped uv, back to origial uv to help do the uvd->xyz operation
    :return:
    """
    N, T, V,_ = cropped_uv.size()
    uv = (cropped_uv+1)*(bb_box[:, 2:].view(N, 1, 1, 2)/2.0)+bb_box[:, 0:2].view(N, 1, 1, 2)
    return uv

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def normalize_data( data, data_mean, data_std, dim_to_use, actions,dim=3):
  data_out = {}
  nactions = len(actions)
  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev + 1E-8)

  return data_out


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]
  

def define_actions_humaneva( action ):
  actions = ["Walking", "Jog", "Box"]

  return actions

def define_actions_3dhp( action, train ):
  if train:
    actions = ["Seq1", "Seq2"]
  else:
    actions = ["Seq1"]

    return actions

def gaussian(input_2D):
    gaussian_scale = 5 / 500

    input_2D = input_2D.cpu().numpy()
    nosie = np.random.normal(0, gaussian_scale, input_2D.shape)
    input_2D = input_2D + nosie
    input_2D = torch.from_numpy(input_2D.astype('float32'))
    input_2D = input_2D.cuda()

    return input_2D


def gettime():
    time_now = time.localtime()

    month = int(time.strftime("%m", time_now))
    day = int(time.strftime("%d", time_now))
    hour = int(time.strftime("%H", time_now))
    minute = int(time.strftime("%M", time_now))

    return month, day, hour, minute

def gaussian_loss_opt(M, S, target): 
    # M:mu, S:s;  M.shape,s.shape:[B,1,17,3]
    # target:Ground Truth 3D
    assert M.shape == target.shape
    S = S.squeeze(-1)
    S = torch.clamp(S, -3)
    # return torch.mean((torch.norm(target - M, p=2, dim=-1) * torch.exp(S)))
    return torch.mean(torch.norm(target - M, p=2, dim=-1) * torch.exp(-S))

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3  #  B,J,3
    assert len(camera_params.shape) == 2  # camera_params:[B,1,9]
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7] # B,1,3
    p = camera_params[..., 7:] # B,1,2

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1) # B,J,2
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True) # B, J, 1

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,   # B,J,1
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True) # B,J,1

    XXX = XX * (radial + tan) + p * r2 # B,J,2

    return f * XXX + c