import os
import random
import time
import pickle
import glob

import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import utils

from torch.distributed import ReduceOp
from dataloader.data_load import PlanningDataset
from model import diffusion_act
from model import temporal_act

from utils.args import get_args
from utils.training_act import Trainer
from utils.eval import validate
from model.helpers import AverageMeter
import numpy as np

def accuracy2(output, target, topk=(1,), max_traj_len=0):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()        
        comparison = torch.cat((pred.view(-1, max_traj_len),target.view(-1, max_traj_len)), axis=1).cpu().numpy()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_a = correct[:1].view(-1, max_traj_len)
        correct_a0 = correct_a[:, 0].reshape(-1).float().mean().mul_(100.0)
        correct_aT = correct_a[:, -1].reshape(-1).float().mean().mul_(100.0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]

        # Success Rate
        trajectory_success = torch.all(correct_1.view(correct_1.shape[1] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        _, pred_token = output.topk(1, 1, True, True)
        pred_inst = pred_token.view(correct_1.shape[1], -1)
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[1], -1)
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            pred_inst_set.add(tuple(pred_inst[i].tolist()))
            target_inst_set.add(tuple(target_inst[i].tolist()))
        MIoU1 = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(pred_inst_set.union(target_inst_set))

        batch_size = batch_size // max_traj_len
        pred_inst = pred_token.view(batch_size, -1)  # [bs, T]
        pred_inst_set = set()
        target_inst = target.view(batch_size, -1)  # [bs, T]
        target_inst_set = set()
        MIoU_sum = 0
        for i in range(pred_inst.shape[0]):
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())
            MIoU_current = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(
                pred_inst_set.union(target_inst_set))
            MIoU_sum += MIoU_current
            pred_inst_set.clear()
            target_inst_set.clear()

        MIoU2 = MIoU_sum / batch_size
        return res[0], trajectory_success_rate, MIoU1, MIoU2, correct_a0, correct_aT, comparison

def get_noise_mask(action_label, args, img_tensors, act_emd):        
        output_act_emb = torch.randn_like(img_tensors).cuda()
        act_emd = act_emd.cuda()
        
        if args.mask_type == 'single_add':
            for i in range(action_label.shape[0]):
                for j in range(action_label.shape[1]):
                    output_act_emb[i][j][args.class_dim+args.action_dim:args.class_dim+args.action_dim+512] = act_emd[action_label[i][j]]
            return output_act_emb.cuda()
            
        if args.mask_type == 'multi_add':
            for i in range(action_label.shape[0]):
                for j in range(action_label.shape[1]):
                    if j==0:
                        output_act_emb[i][j][args.class_dim+args.action_dim:args.class_dim+args.action_dim+512] = act_emd[action_label[i][j]]
                    else:
                        output_act_emb[i][j][args.class_dim+args.action_dim:args.class_dim+args.action_dim+512] = output_act_emb[i][j-1][args.class_dim+args.action_dim:args.class_dim+args.action_dim+512] + act_emd[action_label[i][j]]
                        
            return output_act_emb.cuda()
                    
def test(val_loader, model, args, act_emb):
    model.eval()
    acc_top1 = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU1_meter = AverageMeter()
    MIoU2_meter = AverageMeter()

    A0_acc = AverageMeter()
    AT_acc = AverageMeter()
    pred_gt_total = []

    for i_batch, sample_batch in enumerate(val_loader):
        # compute output
        global_img_tensors = sample_batch[0].cuda().contiguous()
        video_label = sample_batch[1].cuda()
        batch_size_current, T = video_label.size()
        task_class = sample_batch[2].view(-1).cuda()
        cond = {}

        with torch.no_grad():
            cond[0] = global_img_tensors[:, 0, :].float()
            cond[T - 1] = global_img_tensors[:, -1, :].float()

            task_onehot = torch.zeros((task_class.size(0), args.class_dim))
            # [bs*T, ac_dim]
            ind = torch.arange(0, len(task_class))
            task_onehot[ind, task_class] = 1.
            task_onehot = task_onehot.cuda()
            temp = task_onehot.unsqueeze(1)
            task_class_ = temp.repeat(1, T, 1)  # [bs, T, args.class_dim]
            cond['task'] = task_class_

            video_label_reshaped = video_label.view(-1)
            img_tensors = torch.zeros((batch_size_current, T, args.class_dim + args.action_dim + args.observation_dim))
            img_tensors[:, 0, args.class_dim+args.action_dim:] = global_img_tensors[:, 0, :]
            img_tensors[:, -1, args.class_dim+args.action_dim:] = global_img_tensors[:, -1, :]            
            img_tensors[:, :, :args.class_dim] = task_class_
            
            noise = get_noise_mask(sample_batch[1], args, img_tensors, act_emb)
            
            output = model(cond, noise, task_class, if_jump=True, if_avg_mask=args.infer_avg_mask)

            actions_pred = output.contiguous()
            actions_pred = actions_pred[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
            actions_pred = actions_pred.view(-1, args.action_dim)

            acc1, trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc, pred_gt = accuracy2(actions_pred.cpu(), video_label_reshaped.cpu(), topk=(1,), max_traj_len=args.horizon)
            pred_gt_total.append(pred_gt)

        acc_top1.update(acc1.item(), batch_size_current)
        trajectory_success_rate_meter.update(trajectory_success_rate.item(), batch_size_current)
        MIoU1_meter.update(MIoU1, batch_size_current)
        MIoU2_meter.update(MIoU2, batch_size_current)
        A0_acc.update(a0_acc, batch_size_current)
        AT_acc.update(aT_acc, batch_size_current)
    
    np.savetxt("pred_gt_"+args.dataset+str(args.horizon)+".csv", np.concatenate(pred_gt_total), delimiter=",")
    return torch.tensor(acc_top1.avg), \
           torch.tensor(trajectory_success_rate_meter.avg), \
           torch.tensor(MIoU1_meter.avg), torch.tensor(MIoU2_meter.avg), \
           torch.tensor(A0_acc.avg), torch.tensor(AT_acc.avg)

def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node:', ngpus_per_node)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Test data loading code
    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_sampler.shuffle = False
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )
    
    # read action embeddings
    if args.dataset == 'crosstask' or args.dataset == 'NIV':
        with open(args.act_emb_path, 'rb') as f:
            act_emb = pickle.load(f)
        
        ordered_act = dict(sorted(act_emb.items()))
        feature = []
        for i in ordered_act.keys():
            feature.append(ordered_act[i])
	
        feature = np.array(feature)
        act_emb = torch.tensor(feature)
    
    if args.dataset == 'coin':
        with open(args.act_emb_path, 'rb') as f:
            act_emb = pickle.load(f)
        ordered_act = dict(sorted(act_emb['steps_to_embeddings'].items()))
        feature = []
        for i in ordered_act.keys():
            feature.append(ordered_act[i])
	
        feature = np.array(feature)
        act_emb = torch.tensor(feature)

    # create model            
    if args.dataset=='NIV':
        if args.attn=='NoAttention':
            temporal_model = temporal_act.TemporalUnetNoAttn(args.action_dim + args.observation_dim + args.class_dim, args.action_dim, dim=256, dim_mults=(1, 2, 4, 8), )
        if args.attn=='WithAttention':
            temporal_model = temporal_act.TemporalUnet(args.action_dim + args.observation_dim + args.class_dim, args.action_dim, dim=256, dim_mults=(1, 2, 4, 8), )
    else:
        if args.attn=='NoAttention':
            temporal_model = temporal_act.TemporalUnetNoAttn(args.action_dim + args.observation_dim + args.class_dim, args.action_dim, dim=512, dim_mults=(1, 2, 4), )
        if args.attn=='WithAttention':
            temporal_model = temporal_act.TemporalUnet(args.action_dim + args.observation_dim + args.class_dim, args.action_dim, dim=512, dim_mults=(1, 2, 4), )
    

    diffusion_model = diffusion_act.GaussianDiffusion(
        temporal_model, args.horizon, args.observation_dim, args.action_dim, args.class_dim, args.n_diffusion_steps,
        loss_type='Weighted_MSE', clip_denoised=True,)

    model = Trainer(diffusion_model, None, args.ema_decay, args.lr, args.gradient_accumulate_every, args.step_start_ema, args.update_ema_every, args.log_freq, act_emb)

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            model.model.cuda(args.gpu)
            model.ema_model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[args.gpu], find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(model.ema_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.model.cuda()
            model.ema_model.cuda()
            model.model = torch.nn.parallel.DistributedDataParallel(model.model, find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(model.ema_model, find_unused_parameters=True)

    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.ema_model = torch.nn.DataParallel(model.ema_model).cuda()

    if args.resume:
        checkpoint_path = args.checkpoint_diff
        
        if checkpoint_path:
            print("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"], strict=True)
            model.ema_model.load_state_dict(checkpoint["ema"], strict=True)
            model.step = checkpoint["step"]
        else:
            assert 0

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    time_start = time.time()
    acc_top1_reduced_sum = []
    trajectory_success_rate_meter_reduced_sum = []
    MIoU1_meter_reduced_sum = []
    MIoU2_meter_reduced_sum = []
    acc_a0_reduced_sum = []
    acc_aT_reduced_sum = []
    test_times = 10

    for epoch in range(0, test_times):
        tmp = epoch
        random.seed(tmp)
        np.random.seed(tmp)
        torch.manual_seed(tmp)
        torch.cuda.manual_seed_all(tmp)

        acc_top1, trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, acc_a0, acc_aT = test(test_loader, model.ema_model, args, act_emb)

        acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
        trajectory_success_rate_meter_reduced = reduce_tensor(trajectory_success_rate_meter.cuda()).item()
        MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
        MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
        acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
        acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

        acc_top1_reduced_sum.append(acc_top1_reduced)
        trajectory_success_rate_meter_reduced_sum.append(trajectory_success_rate_meter_reduced)
        MIoU1_meter_reduced_sum.append(MIoU1_meter_reduced)
        MIoU2_meter_reduced_sum.append(MIoU2_meter_reduced)
        acc_a0_reduced_sum.append(acc_a0_reduced)
        acc_aT_reduced_sum.append(acc_aT_reduced)

    if args.rank == 0:
        time_end = time.time()
        print('time: ', time_end - time_start)
        print('-----------------Mean&Var-----------------------')
        print('Val/EpochAcc@1', sum(acc_top1_reduced_sum) / test_times, np.var(acc_top1_reduced_sum))
        print('Val/Traj_Success_Rate', sum(trajectory_success_rate_meter_reduced_sum) / test_times, np.var(trajectory_success_rate_meter_reduced_sum))
        print('Val/MIoU1', sum(MIoU1_meter_reduced_sum) / test_times, np.var(MIoU1_meter_reduced_sum))
        print('Val/MIoU2', sum(MIoU2_meter_reduced_sum) / test_times, np.var(MIoU2_meter_reduced_sum))
        print('Val/acc_a0', sum(acc_a0_reduced_sum) / test_times, np.var(acc_a0_reduced_sum))
        print('Val/acc_aT', sum(acc_aT_reduced_sum) / test_times, np.var(acc_aT_reduced_sum))
        print('-----------------First-----------------------')
        print('Val/EpochAcc@1', acc_top1_reduced_sum[0] )
        print('Val/Traj_Success_Rate', trajectory_success_rate_meter_reduced_sum[0] ) 
        print('Val/MIoU2', MIoU2_meter_reduced_sum[0] )
        print('-----------------Max SR-----------------------')
        max_v =  max(trajectory_success_rate_meter_reduced_sum)
        max_ind = trajectory_success_rate_meter_reduced_sum.index(max_v)
        print('Val/EpochAcc@1', acc_top1_reduced_sum[max_ind])
        print('Val/Traj_Success_Rate', max(trajectory_success_rate_meter_reduced_sum))
        print('Val/MIoU2', MIoU2_meter_reduced_sum[max_ind])
        print('-----------------Max mACC-----------------------')
        max_v =  max(acc_top1_reduced_sum)
        max_ind = acc_top1_reduced_sum.index(max_v)
        print('Val/EpochAcc@1', acc_top1_reduced_sum[max_ind], max_ind)
        print('Val/Traj_Success_Rate', trajectory_success_rate_meter_reduced_sum[max_ind], max_ind)        
        print('Val/MIoU2', MIoU2_meter_reduced_sum[max_ind], max_ind)
        print('-----------------Max mIoU-----------------------')
        max_v =  max(MIoU2_meter_reduced_sum)
        max_ind = MIoU2_meter_reduced_sum.index(max_v)
        print('Val/EpochAcc@1', acc_top1_reduced_sum[max_ind], max_ind)        
        print('Val/Traj_Success_Rate', trajectory_success_rate_meter_reduced_sum[max_ind], max_ind)       
        print('Val/MIoU2', MIoU2_meter_reduced_sum[max_ind], max_ind)

        


if __name__ == "__main__":
    main()
