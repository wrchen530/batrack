import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange, repeat
import pypose as pp

from model.utils import bilinear_sample2d, align_depth_maps
from model.geomeotry import iproj


class RefineNet(nn.Module):
    def __init__(self, device, result_path, grid_size=4, pw_break=20, verbose=True, scale_mode='exp', align_depth=False, loss_weight_dict=None, refine_intrinsics=False, alpha=0.5, scale_smoothness_weight=0.1, scale_smoothness_mode='l2'):
        super().__init__()

        self.K_scale = 20
        self.device = device
        self.align_depth = align_depth
        self._init_from_ba(result_path)
        self.verbose = verbose
        
        self.loss_weight_dict = loss_weight_dict
        self.alpha = alpha
        self.scale_smoothness_weight = scale_smoothness_weight
        self.scale_smoothness_mode = scale_smoothness_mode

        self.result_path = result_path
        if isinstance(grid_size, (list, tuple)):
            grid_h, grid_w = grid_size
        else:
            grid_h = grid_w = grid_size
        self.grid_size = (grid_h, grid_w)

        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.scale_mode = scale_mode

        self.trajs_scales = nn.Parameter(torch.ones([self.T, self.N, self.S_local], dtype=torch.float32, device=self.device))
        self.frame_scales_ = nn.Parameter(torch.ones([self.T, grid_h, grid_w], dtype=torch.float32, device=self.device))
        self.frame_shifts_ = torch.zeros([self.T], dtype=torch.float32, device=self.device)

        self.pose = pp.Parameter(self.pose_init)
        self.refine_intrinsics = refine_intrinsics
        self.K = nn.Parameter(self.K_init)

    @property
    def frame_shifts(self):
        return self.frame_shifts_
    
    def _init_from_ba(self, result_path):
        results = pickle.load(open(result_path, 'rb'))
        self.results = results
        self.dmaps_raw = results['dmaps']
        
        self.dmaps_aligned = align_depth_maps(results['dmaps'])

        self.cams_T_world = results['cams_T_world'] # B, 4, 4
        self.pose_init = pp.mat2SE3(self.cams_T_world)

        if self.align_depth:
            dmaps = torch.from_numpy(self.dmaps_aligned).to(self.device)
        else:
            dmaps = torch.from_numpy(results['dmaps']).to(self.device)

        self.dmaps = rearrange(dmaps, 't h w c -> t c h w')
                          
        self.trajs_2d_disp = torch.from_numpy(results['trajs_2d_disp']).to(self.device)           # T, N, S_local, 3
        self.grid_query_frames = torch.from_numpy(results['grid_query_frames']).to(self.device)    # T
        self.trajs_valid = torch.from_numpy(results['trajs_valid']).to(self.device)            # T, N
        self.trajs_static = torch.from_numpy(results['trajs_static']).to(self.device)           # T, N, S_local
        self.trajs_vis = torch.from_numpy(results['trajs_vis']).to(self.device)                   # T, N, S_local

        self.intrinsics_raw = torch.from_numpy(results['intrinsics']).to(self.device)                 # T, 4
        self.K_init = torch.median(self.intrinsics_raw, dim=0)[0] / self.K_scale
 
        self.trajs_disp = self.trajs_2d_disp[..., 2]
        self.trajs_2d = self.trajs_2d_disp[..., :2]


        self.T, self.N, self.S_local, _ = self.trajs_2d_disp.shape
        self.H, self.W = self.dmaps.shape[-2:]

        ii = torch.arange(self.T)

        mid_idx = self.S_local // 2
        jj = ii.view(-1, 1) + torch.arange(self.S_local)[None] - mid_idx
        

        self.ii = repeat(ii, 't -> t s', s=self.S_local).to(self.device) 
        self.jj = jj.to(self.device)

        trajs_disp_mono_all = []
        
        for t in range(self.T):
            jj = self.jj[t].clamp(0, self.T-1)
            dmaps_ = self.dmaps[jj] # S_local, 1, H, W
            trajs_2d_ = self.trajs_2d[t]
            trajs_2d_ = rearrange(trajs_2d_, 'n s c -> s n c')

            trajs_depth_mono_ = bilinear_sample2d(dmaps_, trajs_2d_[..., 0], trajs_2d_[..., 1])
            trajs_depth_mono_ = rearrange(trajs_depth_mono_, 's c n -> n s c')
            trajs_disp_mono_ = 1.0 / trajs_depth_mono_.clamp(1e-2)
            trajs_disp_mono_all.append(trajs_disp_mono_[...,0])

        self.trajs_disp_mono = torch.stack(trajs_disp_mono_all, dim=0)

        traj_static_ = rearrange(self.trajs_static, 't n s -> t s n')
        self.trajs_static_mat = traj_static_.unsqueeze(3) @ traj_static_.unsqueeze(2)

        traj_vis_ = rearrange(self.trajs_vis, 't n s -> t s n')
        self.trajs_vis_mat = traj_vis_.unsqueeze(3) @ traj_vis_.unsqueeze(2)

        trajs_disp_mono_mask = (self.trajs_disp_mono > 1e-2).float()
        trajs_disp_mono_mask_ = rearrange(trajs_disp_mono_mask, 't n s -> t s n')
        self.trajs_disp_mono_mask_mat = trajs_disp_mono_mask_.unsqueeze(3) @ trajs_disp_mono_mask_.unsqueeze(2)


    def get_trajs_scales(self):
        trajs_scales = self.trajs_scales
        if self.norm_pw_scale:
            trajs_scales = trajs_scales - trajs_scales.mean(dim=1, keepdim=True)
        return (trajs_scales / self.pw_break).exp()
    

    @property
    def intrinsics(self):
        if self.refine_intrinsics:
            return repeat(self.K, 'c -> t c', t=self.T) * self.K_scale
        else:
            return self.intrinsics_raw
        
    @property
    def frame_scales(self):
        if self.scale_mode == 'exp':
            return (self.frame_scales_ / 10.0).exp()
        elif self.scale_mode == 'raw':
            return self.frame_scales_
        else:
            raise ValueError(f'bad scale mode {self.scale_mode=}')
    
    @property
    def trajs_disp_mono_scaled(self):
        return self.get_frame_scaled_depth()
    
    def get_frame_scaled_depth(self):
        frame_scales = self.frame_scales
        frame_scales = rearrange(frame_scales, 't h w -> t 1 h w')

        frame_shifts = self.frame_shifts

        # sample scales at 2d points
        frame_scales_2d = []
        frame_shifts_2d = []
        for t in range(self.T):
            trajs_2d_ = self.trajs_2d[t]
            trajs_2d_ = rearrange(trajs_2d_, 'n s c -> s n c')
            grid_xy = trajs_2d_.clone()
            grid_xy[...,0] = grid_xy[...,0] / (self.W - 1) * 2 - 1.0
            grid_xy[...,1] = grid_xy[...,1] / (self.H - 1) * 2 - 1.0

            jj = self.jj[t].clamp(0, self.T-1)
            frame_scales_t = frame_scales[jj]
            frame_scales_t_2d = F.grid_sample(frame_scales_t, grid_xy[:,:,None], align_corners=True)
            frame_scales_t_2d = rearrange(frame_scales_t_2d[...,0], 's c n -> n s c').squeeze(2)
            frame_shifts_2d.append(frame_shifts[jj])
            frame_scales_2d.append(frame_scales_t_2d)
        frame_scales_2d = torch.stack(frame_scales_2d, dim=0)
        frame_shifts_2d = torch.stack(frame_shifts_2d, dim=0)

        trajs_disp_mono_scaled = self.trajs_disp_mono * frame_scales_2d + frame_shifts_2d[:,None]
        return trajs_disp_mono_scaled
    
    def dist(self, a, b, weight, mode='l1'):
        if mode == 'l1':
            return ((a - b).norm(dim=-1) * weight)
        elif mode == 'l2':
            return ((a - b).norm(dim=-1) ** 2 * weight)
        elif mode == 'huber':
            return F.smooth_l1_loss(a, b, reduction='none')[...,0] * weight
        elif mode == 'min_max':
            a = a.clamp(1e-2)
            b = b.clamp(1e-2)
            return (torch.max(a,b) / torch.min(a,b)) - 1

    def pairwise_dist(self, x):
        """
        a : B, N, C
        """
        a = x.unsqueeze(2)
        b = x.unsqueeze(1)
        return (a - b).norm(dim=-1)
    

    def inter_frame_loss(self):
        trajs_disp_mono_scaled = self.get_frame_scaled_depth()

        trajs_2d_disp_mono_scaled = torch.cat([self.trajs_2d, trajs_disp_mono_scaled[...,None]], dim=-1)
        trajs_2d_disp_mono_scaled = rearrange(trajs_2d_disp_mono_scaled, 't n s c -> t s n c')
        loss = 0.0

        mid_idx = self.S_local // 2 
        for i in self.grid_query_frames:
            jj = self.jj[i]
            intrinsics = self.intrinsics[jj.clamp(0, self.T-1)] 
            trajs_3d_ = iproj(trajs_2d_disp_mono_scaled[i], intrinsics)
            pair_dist = self.pairwise_dist(trajs_3d_)
            pair_dist_diff = (pair_dist - pair_dist[mid_idx]).abs()

            t_mask = (jj >= 0) & (jj < self.T)
            vis_mask = self.trajs_vis_mat[i] > 0.5
            static_mask = self.trajs_static_mat[i] > 0.5
            disp_mask = self.trajs_disp_mono_mask_mat[i] > 0.5
            mask = t_mask[:, None, None] & vis_mask & static_mask & disp_mask
            
            loss += mask * pair_dist_diff

        loss = loss / self.grid_query_frames.shape[0]
        loss = loss.mean()
        return loss

    def inter_frame_loss_local(self):
        trajs_scales = self.get_trajs_scales()
        trajs_disp_mono_scaled = trajs_scales * self.trajs_disp

        trajs_2d_disp_mono_scaled = torch.cat([self.trajs_2d, trajs_disp_mono_scaled[...,None]], dim=-1)
        trajs_2d_disp_mono_scaled = rearrange(trajs_2d_disp_mono_scaled, 't n s c -> t s n c')
        loss = 0.0

        mid_idx = self.S_local // 2 
        for i in self.grid_query_frames:
            jj = self.jj[i]
            intrinsics = self.intrinsics[jj.clamp(0, self.T-1)] 
            trajs_3d_ = iproj(trajs_2d_disp_mono_scaled[i], intrinsics)
            pair_dist = self.pairwise_dist(trajs_3d_)
            pair_dist_diff = (pair_dist - pair_dist[mid_idx]).abs()

            t_mask = (jj >= 0) & (jj < self.T)
            vis_mask = self.trajs_vis_mat[i] > 0.5
            static_mask = self.trajs_static_mat[i] > 0.5
            disp_mask = self.trajs_disp_mono_mask_mat[i] > 0.5
            mask = t_mask[:, None, None] & vis_mask & static_mask & disp_mask
            
            loss += mask * pair_dist_diff

        loss = loss / self.grid_query_frames.shape[0]
        loss = loss.mean()
        return loss

    def forward(self):
        trajs_scales = self.get_trajs_scales()
        loss = 0
        vis_mask = (self.trajs_vis > 0.9)
        patch_mask = (self.jj >= 0) & (self.jj < self.T)
        patch_mask = repeat(patch_mask, 't s -> t n s', n=self.N)
        
        flow_mask = self.trajs_2d.norm(dim=-1) > 5
        disp_mask = (self.trajs_disp > 1e-2)
        mask = vis_mask & patch_mask & flow_mask & disp_mask
        mask = mask.float()

        algined_trajs = trajs_scales * self.trajs_disp
        trajs_disp_mono_scaled = self.get_frame_scaled_depth()

        loss = self.dist(trajs_disp_mono_scaled[...,None], algined_trajs[...,None], mask, mode='huber')
        loss_spatial = loss[self.grid_query_frames.long()].mean()

        alpha = self.alpha
        if alpha > 0:
            loss_rigid = self.inter_frame_loss()
        else:
            loss_rigid = 0.0

        total_loss = 0.0
        loss_dict = {}

        if self.loss_weight_dict is not None:
            if 'spatial_loss' in self.loss_weight_dict:
                loss_dict['spatial_loss'] = self.loss_weight_dict['spatial_loss'] * loss_spatial
            if 'inter_frame_loss' in self.loss_weight_dict:
                loss_dict['inter_frame_loss'] = self.loss_weight_dict['inter_frame_loss'] * loss_rigid
            if 'cam_smooth_vec_loss' in self.loss_weight_dict:
                cam_smooth_loss = self.cam_smooth_vec_loss()
                loss_dict['cam_smooth_vec_loss'] = self.loss_weight_dict['cam_smooth_vec_loss'] * cam_smooth_loss
            if 'pts_3d_loss' in self.loss_weight_dict:
                pts_3d_loss = self.pts_3d_loss()
                loss_dict['pts_3d_loss'] = self.loss_weight_dict['pts_3d_loss'] * pts_3d_loss
            if 'scale_smoothness_loss' in self.loss_weight_dict:
                scale_smoothness_loss = self.scale_grid_smoothness_loss(mode='l1')
                loss_dict['scale_smoothness_loss'] = self.loss_weight_dict['scale_smoothness_loss'] * scale_smoothness_loss

            total_loss = 0.0
            result_str = ''
            for key, value in loss_dict.items():
                total_loss += value
                result_str += f"{key}: {value.item():.4f}, "
            result_str += f"total: {total_loss.item():.4f}"
            if self.verbose:
                print(result_str)
        else:
            loss_scale_smoothness = self.scale_grid_smoothness_loss(mode='l1') if self.scale_smoothness_weight > 0 else 0.0
            total_loss = loss_spatial + alpha * loss_rigid + self.scale_smoothness_weight * loss_scale_smoothness
            if self.verbose:
                if self.scale_smoothness_weight > 0:
                    print(f"loss_spatial: {loss_spatial.item():.4f}, loss_rigid: {loss_rigid.item() * alpha:.4f}, loss_scale_smoothness: {loss_scale_smoothness.item() * self.scale_smoothness_weight:.4f}, total: {total_loss.item():.4f}")
                else:
                    print(f"loss_spatial: {loss_spatial.item():.4f}, loss_rigid: {loss_rigid.item() * alpha:.4f}, total: {total_loss.item():.4f}")

        return total_loss


    def pts_3d_loss(self):
        mid_idx = self.S_local // 2
        pts_src_2d = self.trajs_2d[:,:,mid_idx]

        trajs_disp_mono_scaled = self.get_frame_scaled_depth()
        pts_src_disp = trajs_disp_mono_scaled[:,:,mid_idx]

        patches = torch.cat([pts_src_2d, pts_src_disp[...,None]], dim=-1)   # S, N, 3
        pts_src_3d = iproj(patches, self.intrinsics).float()

        src2trg_pose = []
        trg_intrinsics = []
        for t in range(self.T):
            jj = self.jj[t].clamp(0, self.T-1)
            jj_src = torch.ones_like(jj) * t
            src2trg = self.pose[jj].Inv() @ self.pose[jj_src]
            src2trg_pose.append(src2trg)
            trg_intrinsics.append(self.intrinsics[jj])
        

        src2trg_pose = torch.stack(src2trg_pose, dim=0)
        src2trg_pose = pp.SE3(src2trg_pose)

        trg_intrinsics = torch.stack(trg_intrinsics, dim=0)

        pts_trg_3d_from_src = src2trg_pose[:, None] @ pts_src_3d[:,:,None]

        patches_trg = torch.cat([self.trajs_2d, trajs_disp_mono_scaled[...,None]], dim=-1)
        pts_trg_3d = iproj(patches_trg, trg_intrinsics[:,None]).float()
        pts_dist = (pts_trg_3d_from_src - pts_trg_3d).norm(dim=-1)

        vis_mask = (self.trajs_vis > 0.9)
        patch_mask = (self.jj >= 0) & (self.jj < self.T)
        patch_mask = repeat(patch_mask, 't s -> t n s', n=self.N)
        disp_mask = (self.trajs_disp > 1e-2)
        static_mask = (self.trajs_static > 0.3)
        mask = vis_mask & patch_mask & disp_mask & static_mask
        mask = mask.float()

        pts_dist_lostt = (pts_dist * mask).mean()
        return pts_dist_lostt

    def cam_smooth_vec_loss(self):
        loss_trans = (self.pose[:-1, :3].tensor() - self.pose[1:, :3].tensor()).norm(dim=-1)
        loss_rot = (self.pose[:-1, 3:].tensor() - self.pose[1:, 3:].tensor()).norm(dim=-1)
        loss_mean = loss_trans.mean() + loss_rot.mean() * 0.3
        return loss_mean
    
    def scale_grid_smoothness_loss(self, mode='l2'):
        """
        Compute smoothness regularization loss for the scale grid.
        Penalizes differences between neighboring scales in spatial dimensions.
        
        Args:
            mode (str): 'l2' for L2 loss, 'l1' for L1 loss, 'huber' for Huber loss
        """
        scales = self.frame_scales  # T, grid_h, grid_w
        
        # Compute horizontal differences (neighboring pixels in width direction)
        diff_h = scales[:, :, :-1] - scales[:, :, 1:]  # T, grid_h, grid_w-1
        
        # Compute vertical differences (neighboring pixels in height direction)
        diff_v = scales[:, :-1, :] - scales[:, 1:, :]  # T, grid_h-1, grid_w
        
        if mode == 'l2':
            loss_h = (diff_h ** 2).mean()
            loss_v = (diff_v ** 2).mean()
        elif mode == 'l1':
            loss_h = diff_h.abs().mean()
            loss_v = diff_v.abs().mean()
        elif mode == 'huber':
            loss_h = F.smooth_l1_loss(diff_h, torch.zeros_like(diff_h), reduction='mean')
            loss_v = F.smooth_l1_loss(diff_v, torch.zeros_like(diff_v), reduction='mean')
        else:
            raise ValueError(f"Unknown smoothness loss mode: {mode}")
        
        # Total smoothness loss
        smoothness_loss = loss_h + loss_v
        return smoothness_loss
    
    def get_results(self):
        
        results = self.results
        dmaps_scaled = self.scaled_dmaps
        results['final_trajs_2d'] = self.trajs_2d.detach().cpu().numpy()
        results['dmaps'] = self.dmaps.detach().cpu().numpy()
        results['dmaps_scaled'] = dmaps_scaled.detach().cpu().numpy()
        results['cams_T_world'] = self.pose.matrix().detach().cpu().numpy()
        results['intrinsics'] = self.intrinsics.detach().cpu().numpy()

        return results  
    


    @property
    def scaled_dmaps(self):
        scales = self.frame_scales
        shifts = self.frame_shifts
        scales = rearrange(scales, 't h w -> t 1 h w')
        scales_maps = F.interpolate(scales, size=(self.H, self.W), mode='bilinear', align_corners=True)
        dmaps = self.dmaps
        dmaps_scaled = dmaps / (scales_maps + shifts.view(-1, 1, 1, 1) * dmaps)
        return dmaps_scaled
    
    @property
    def scales_map(self):
        scales = self.frame_scales
        shifts = self.frame_shifts
        scales = rearrange(scales, 't h w -> t 1 h w')
        scales_dense = F.interpolate(scales, size=(self.H, self.W), mode='bilinear', align_corners=True)

        return 1.0 / scales_dense