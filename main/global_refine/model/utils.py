import os

import torch
import numpy as np
from scipy.optimize import minimize

def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    if len(im.shape) == 5:
        B, N, C, H, W = list(im.shape)
    else:
        B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    if len(im.shape) == 5:
        im_flat = (im.permute(0, 3, 4, 1, 2)).reshape(B * H * W, N, C)
        i_y0_x0 = torch.diagonal(im_flat[idx_y0_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y0_x1 = torch.diagonal(im_flat[idx_y0_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x0 = torch.diagonal(im_flat[idx_y1_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x1 = torch.diagonal(im_flat[idx_y1_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
    else:
        im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
        i_y0_x0 = im_flat[idx_y0_x0.long()]
        i_y0_x1 = im_flat[idx_y0_x1.long()]
        i_y1_x0 = im_flat[idx_y1_x0.long()]
        i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    )
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


def eval_depth(ba_model, depth_min=1e-2, depth_max=1e2, scaling='median', scene_name='all'):
    dmaps_gt = ba_model.results['dmaps_gt'][...,0]
    dmaps_scaled = ba_model.scaled_dmaps[:,0].detach().cpu().numpy()


    mask = (dmaps_gt > 1e-2) & (dmaps_gt < 1e2)
    
    depth_dict = {
        'final': dmaps_scaled
    }

    results = eval_depth_metric(dmaps_gt, depth_dict, mask, exp_name=scene_name, depth_min=depth_min, depth_max=depth_max, scaling=scaling)
    return results


def absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)

    optimizer = torch.optim.Adam([s, t], lr=lr)
    
    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()

def align_with_la2d(predicted_depth, ground_truth_depth, lr=0.01, max_iters=300):
    print("Aligning with LA2D")
    # from numpy as to tensor
    predicted_depth = torch.from_numpy(predicted_depth)
    ground_truth_depth = torch.from_numpy(ground_truth_depth)

    s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
    s, t = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters)
    predicted_depth = s * predicted_depth + t

    # from numpy as to tensor
    predicted_depth = predicted_depth.detach().cpu().numpy()
    return predicted_depth

def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def align_with_lad(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.reshape(-1)
    ground_truth_depth_np = ground_truth_depth.reshape(-1)
    initial_params = [s, t]

    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))

    s, t = result.x

    predicted_depth = s * predicted_depth + t
    return predicted_depth

def align_with_lstsq(predicted_depth, ground_truth_depth):
    print("Aligning with least squares")
    predicted_depth_np = predicted_depth.reshape(-1, 1)
    ground_truth_depth_np = ground_truth_depth.reshape(-1, 1)
    
    # Add a column of ones for the shift term
    A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
    
    # Solve for scale (s) and shift (t) using least squares
    result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
    s, t = result[0][0], result[0][1]

    # Apply scale and shift
    predicted_depth = s * predicted_depth + t
    return predicted_depth

def compute_errors(gt, pred, min_depth, max_depth, scaling='median'):
    """Computation of error metrics between predicted and ground truth depths
    """
    if scaling == 'median':
        ratio = np.median(gt) / np.median(pred)
        pred *= ratio
    elif scaling == 'la2d':
        pred = align_with_la2d(pred, gt)
    elif scaling == 'lad':
        pred = align_with_lad(pred, gt)
    elif scaling == 'lstsq':
        pred = align_with_lstsq(pred, gt)

    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))
    abs_diff_median = np.median(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    mask = np.ones_like(pred)

    return abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3



def print_results(exp_name, results):
    print(f"\n {exp_name}")
    print("\n  {:>10}|".format('depth') +
            ("{:>8} | " *
            8).format("abs_rel", "sq_rel", "log10", "rmse", "rmse_log",
                            "a1", "a2", "a3"))
    for key, value in results.items():
        print(("{:>10} " + "&{: 8.3f}  " * 8).format(key, *value.tolist()) + "\\\\")

def eval_depth_metric(gt_depth, pred_depth_dict, mask, exp_name='', depth_min=1e-2, depth_max=1e2, scaling='median'):
    results = {}
    # only compute loss on valid depth (within range)
    mask_depth = (gt_depth > depth_min) & (gt_depth < depth_max)
    mask_valid = mask_depth & mask

    for key, value in pred_depth_dict.items():
        gt_depth_vis = gt_depth[mask_valid].reshape(-1)
        pred_depth_vis = value[mask_valid].reshape(-1)
        errors_ = compute_errors(gt_depth_vis, pred_depth_vis, min_depth=depth_min, max_depth=depth_max, scaling=scaling)
        
        results[key] = np.array(errors_)

    print_results(exp_name, results)
    return results


def align_depth_maps(depth_maps):
    S, H, W, _ = depth_maps.shape
    aligned_depth_maps = np.zeros_like(depth_maps)
    aligned_depth_maps[0] = depth_maps[0]  # Initialize the first depth map as the reference

    min_overlap_threshold = 100  # Minimum number of overlapping pixels required

    # Loop through each consecutive pair of depth maps to align them incrementally
    for i in range(1, S):
        D_prev = aligned_depth_maps[i - 1, ..., 0]
        D_curr = depth_maps[i, ..., 0]

        # Find the overlapping region using a mask (non-zero values assumed as valid depth)
        mask = (D_prev > 0) & (D_curr > 0)
        overlap_count = np.sum(mask)
        if overlap_count < min_overlap_threshold:
            print(f"Insufficient overlapping region found between depth map {i - 1} and {i} ({overlap_count} pixels). Using previous transformation.")
            aligned_depth_maps[i, ..., 0] = D_curr
            continue

        # Extract overlapping depths
        D_prev_overlap = D_prev[mask]
        D_curr_overlap = D_curr[mask]

        # Solve for scale using median alignment such that the current median matches the median of the past two depth maps
        if i == 1:
            median_prev = np.median(D_prev_overlap)
        else:
            D_past_prev = aligned_depth_maps[i - 2, ..., 0]
            mask_past = (D_past_prev > 0) & (D_prev > 0)
            D_past_prev_overlap = D_past_prev[mask_past]
            D_prev_overlap_combined = np.concatenate((D_past_prev_overlap, D_prev_overlap))
            median_prev = np.median(D_prev_overlap_combined)
        
        median_curr = np.median(D_curr_overlap)
        scale = median_prev / median_curr

        # Apply the transformation to the current depth map
        aligned_depth_maps[i, ..., 0] = scale * D_curr

        # Calculate and print the alignment loss
        loss = np.mean(np.abs((scale * D_prev_overlap) - D_curr_overlap))
        # print(f"Alignment loss after aligning depth map {i - 1} and {i}: {loss}")

    return aligned_depth_maps

