import torch
import tqdm
import numpy as np

def cosine_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2


def linear_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_start + (lr_end - lr_start) * t


def adjust_learning_rate_by_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr


def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-6, fixed_pose=False, fixed_K=False):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print('Global alignement - optimizing for:')
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    params_op = [
        {'params': net.trajs_scales, 'lr':lr},
        {'params': net.frame_scales_, 'lr':lr},
    ]
    if net.frame_shifts_.requires_grad:
        params_op.append({'params': net.frame_shifts_, 'lr':lr*0.3})
    if not fixed_pose:
        params_op.append({'params': net.pose, 'lr': 1e-2})
    if not fixed_K:
        params_op.append({'params': net.K, 'lr': 1e-2})
    optimizer = torch.optim.Adam(params_op, lr=lr, betas=(0.9, 0.9))


    

    loss = float('inf')
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule)
                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)
    return loss

def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    """Compute one iteration of global alignment
    """
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()
    loss = net()
    loss.backward()
    optimizer.step()

    return float(loss), lr
