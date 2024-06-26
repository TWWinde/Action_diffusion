import torch


def accuracy(output, target, topk=(1,), max_traj_len=0):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # [k, bs*T]
        #print(pred, target.view(1, -1).expand_as(pred))
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [k, bs*T]

        correct_a = correct[:1].view(-1, max_traj_len)  # [bs, T]
        correct_a0 = correct_a[:, 0].reshape(-1).float().mean().mul_(100.0)
        correct_aT = correct_a[:, -1].reshape(-1).float().mean().mul_(100.0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]     # (1, bs*T)

        # Success Rate
        trajectory_success = torch.all(correct_1.view(correct_1.shape[1] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        _, pred_token = output.topk(1, 1, True, True)  # [bs*T, 1]
        pred_inst = pred_token.view(correct_1.shape[1], -1)  # [bs*T, 1]
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[1], -1)  # [bs*T, 1]
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            # print(pred_inst[i], target_inst[i])
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
            # print(pred_inst[i], target_inst[i])
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())
            MIoU_current = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(
                pred_inst_set.union(target_inst_set))
            MIoU_sum += MIoU_current
            pred_inst_set.clear()
            target_inst_set.clear()

        MIoU2 = MIoU_sum / batch_size
        return res, trajectory_success_rate, MIoU1, MIoU2, correct_a0, correct_aT


def similarity_score(pred, act_emb, metric='cos'):
    # pred shape: [bs*t, 512] 512 is action embedding shape
    sim_score = torch.zeros((pred.shape[0], act_emb.shape[0])).cuda()
    if metric == 'cos':
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        #print(pred[0].shape, sim_score.shape, act_emb[0].shape)
        for i in range(sim_score.shape[0]):            
            for j in range(act_emb.shape[0]):
                sim_score[i][j] = cos(pred[i],act_emb[j].cuda())
        
        sim_score = torch.abs(sim_score)
        print('sim_score', sim_score, torch.max(sim_score, 1))
    return sim_score.cpu()
