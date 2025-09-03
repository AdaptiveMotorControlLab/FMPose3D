import os
import glob
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from common.arguments import opts as parse_args
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
import time

args = parse_args().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
exec('from model.' + args.model + ' import Model as CFM')

# wandb logging (assumed available)
import wandb
WANDB_AVAILABLE = True

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)

def step(split, args, actions, dataLoader, model, optimizer=None, epoch=None):

    loss_all = {'loss': AccumLoss()}

    action_error_sum = define_error_list(actions)
    
    model_3d = model['CFM']
    if split == 'train':
        model_3d.train()
    else:
        model_3d.eval()

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split =='train':
            # if i>20:
            #     continue
            # Conditional Flow Matching training
            # gt_3D, input_2D shape: (B,F,J,C)
            x0 = torch.randn_like(gt_3D)
            B = gt_3D.size(0)
            # t on correct device/dtype and broadcastable: (B,1,1,1)
            t = torch.rand(B, 1, 1, 1, device=gt_3D.device, dtype=gt_3D.dtype)
            y_t = (1.0 - t) * x0 + t * gt_3D
            v_target = gt_3D - x0
            v_pred = model_3d(input_2D, y_t, t)
            
        else:
            # When test_augmentation=True, input_2D has an extra aug dimension: (B,2,F,J,2)
            # For now, use the first view to keep shapes consistent with the model
            input_2D = input_2D[:, 0]
            # Simple Euler sampler for CFM at test time
            # Integrate from noise (t=0) to t=1 using learned velocity field
            steps = getattr(args, 'sample_steps', 16)
            dt = 1.0 / steps
            y = torch.randn_like(gt_3D)
            for s in range(steps):
                t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                v_s = model_3d(input_2D, y, t_s)
                y = y + dt * v_s
            output_3D = y

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            loss = ((v_pred - v_target)**2).mean()
            
            # loss_p1 = mpjpe_cal(output_3D, out_target.clone())
            N = input_2D.size(0)
            
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            if WANDB_AVAILABLE:
                # log per-batch training loss
                wandb.log({'train_loss': float(loss.detach().cpu().item()), 'epoch': epoch if epoch is not None else -1})


            # loss = loss_p1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            
            output_3D = output_3D[:, args.pad].unsqueeze(1) 
            output_3D[:, :, 0, :] = 0
            test_p1 = mpjpe_cal(output_3D, gt_3D)
            
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, args.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg

    elif split == 'test':
        mpjpe_p1, p2 = print_error(args.dataset, action_error_sum, args.train)
        return mpjpe_p1, p2

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime
    
    if WANDB_AVAILABLE:
        wandb.init(project=getattr(args, 'wandb_project', 'Pose3DCFM'),
                   name=f"CFM_{logtime}",
                   config={k: getattr(args, k) for k in dir(args) if not k.startswith('_')})
     
    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = './debug/' + logtime
        else:
            args.checkpoint = './checkpoint/' + logtime
    
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        # backup files
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join( args.checkpoint, args.create_time + "_" + file_name))
        shutil.copyfile(src="common/arguments.py", dst = os.path.join(args.checkpoint, args.create_time + "_arguments.py"))
        shutil.copyfile(src="model/model_CFM.py", dst = os.path.join(args.checkpoint, args.create_time + "_model_CFM.py"))
        shutil.copyfile(src="common/utils.py", dst = os.path.join(args.checkpoint, args.create_time + "_utils.py"))
        shutil.copyfile(src="run.sh", dst = os.path.join(args.checkpoint, args.filename+"_run.sh"))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)
             
        arguments = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(arguments.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

    root_path = args.root_path
    dataset_path = root_path + 'data_3d_' + args.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    if args.train:
        train_data = Fusion(opt=args, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    if args.test:
        test_data = Fusion(opt=args, train=False, dataset=dataset, root_path =root_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = {}
    model['CFM'] = CFM(args).cuda()

    
    if args.reload:
        model_dict = model['CFM'].state_dict()
        # model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]
        model_path = glob.glob(os.path.join(args.previous_dir, '*.pth'))[0]
        # model_path = "./pre_trained_model/IGANet_8_4834.pth"
        print(model_path)
        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['CFM'].load_state_dict(model_dict)
        print("Load model Successfully!")

    all_param = []
    all_paramters = 0
    lr = args.lr
    all_param += list(model['CFM'].parameters())
    print(all_paramters)
    logging.info(all_paramters)
    optimizer = optim.Adam(all_param, lr=args.lr, amsgrad=True)
    
    starttime = datetime.datetime.now()
    best_epoch = 0
    
    for epoch in range(1, args.nepoch):
        if args.train:
            loss = train(args, actions, train_dataloader, model, optimizer, epoch)
            if WANDB_AVAILABLE:
                wandb.log({'train_loss_epoch': float(loss), 'epoch': epoch})
        p1, p2 = val(args, actions, test_dataloader, model)
        if WANDB_AVAILABLE:
            log_data = {'test_p1': p1, 'epoch': epoch}
            wandb.log(log_data)
        
        if args.train:
            data_threshold = p1
            if args.train and data_threshold < args.previous_best_threshold: 
                args.previous_name = save_model(args.previous_name, args.checkpoint, epoch, data_threshold, model['CFM'], "CFM") 
                args.previous_best_threshold = data_threshold
                best_epoch = epoch
                
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.4f, p2: %.4f' % (epoch, lr, loss, p1, p2))
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.4f, p2: %.4f' % (epoch, lr, loss, p1, p2))
        else:        
            print('p1: %.4f, p2: %.4f' % (p1, p2))
            logging.info('p1: %.4f, p2: %.4f' % (p1, p2))
            break
                
        # if epoch % opt.large_decay_epoch == 0: 
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= opt.lr_decay_large
        #         lr *= opt.lr_decay_large
        # else:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
            lr *= args.lr_decay

    endtime = datetime.datetime.now()   
    a = (endtime - starttime).seconds
    h = a//3600
    mins = (a-3600*h)//60
    s = a-3600*h-mins*60
    
    print("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    logging.info("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    print(h,"h",mins,"mins", s,"s")
    logging.info('training time: %dh,%dmin%ds' % (h, mins, s))
    print(args.checkpoint)
    logging.info(args.checkpoint)
    if WANDB_AVAILABLE:
        wandb.finish()