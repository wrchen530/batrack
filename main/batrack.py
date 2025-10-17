import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

from main.backend import altcorr, lietorch
from main.backend.lietorch import SE3
from main.backend import projective_ops as pops
from main.backend.ba import BA_rgbd_droid
from main.slam_visualizer import LEAPVisualizer
from main.frontend.md_tracker import MDTracker
from main.frontend.core.model_utils import (bilinear_sample2d, smart_cat)

def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)

def coords_grid_with_index(d, **kwargs):
    """ coordinate grid with frame index"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index
    

class BATRACK:
    def __init__(self, cfg, ht=480, wd=640):

        self.cfg = cfg
        self.load_weights() 
        self.ht = ht
        self.wd = wd
        self.P = 1      # point tracking: patch_size = 1
        self.S = cfg.model.S
        self.is_initialized = False
        self.enable_timing = False
        self.pred_back = cfg.pred_back if 'pred_back' in cfg else None

        self.use_keyframe = cfg.slam.use_keyframe if 'use_keyframe' in cfg.slam else False
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.slam.PATCHES_PER_FRAME
        self.N = self.cfg.slam.BUFFER_SIZE
        
                
        self.S_model = cfg.model.S
        self.S_slam = cfg.slam.S_slam       # tracked window
        self.S = cfg.slam.S_slam       
        self.kf_stride = cfg.slam.kf_stride
        self.interp_shape = (384, 512)

        self.S_local = self.S_slam * 2 - 1

        # dummy image for visualization
        self.tlist = []
        self.counter = 0
        
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")  
        

        self.patches_ = torch.zeros(self.N, self.M, 3, 1, 1, dtype=torch.float, device="cuda")
        self.patches_local_ = torch.zeros(self.N, self.M, self.S_local, 3, dtype=torch.float, device="cuda")
        
        self.patches_local_monodisp_ = torch.zeros(self.N, self.M, self.S_local, 1, dtype=torch.float, device="cuda")
        self.patches_local_vis_ = torch.zeros(self.N, self.M, self.S_local, 1, dtype=torch.float, device="cuda")
        self.patches_local_static_ = torch.ones(self.N, self.M, self.S_local, 1, dtype=torch.float, device="cuda")
        self.patches_local_weights_ = torch.zeros(self.N, self.M, self.S_local, 1, dtype=torch.float, device="cuda")

        self.patches_monodisp_ = torch.zeros(self.N, self.M, 1, dtype=torch.float, device="cuda")

        self.trajs_3d_world_ = torch.zeros(self.N, self.M, self.S_local, 3, dtype=torch.float, device="cuda")
        


        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")
        
        self.patches_valid_ = torch.zeros(self.N, self.M, dtype=torch.float, device="cuda")
        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
    
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.targets_3d = torch.zeros(1, 0, 3 , device="cuda")
        self.weights = torch.zeros(1, 0, 2 , device="cuda")
        self.weights_pose = torch.zeros(1, 0, 2 , device="cuda")

        # initialize poses to identity matrix, xyzw
        self.poses_[:,6] = 1.0
        
        self.local_window = []
        self.local_window_depth = []
        
        # store relative poses for removed frames
        self.delta = {}
    
        self.viewer = None

        # cache 
        self.cache_window = []
        self.invalid_frames = []


        save_dir = f"{cfg.data.savedir}/{cfg.data.name}"

        self.use_forward = cfg.slam.use_forward if 'use_forward' in cfg.slam else True
        self.use_backward = cfg.slam.use_backward if 'use_backward' in cfg.slam else True
        
        self.visualizer = LEAPVisualizer(cfg, save_dir=save_dir)
    
    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)
    
    
    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, self.P, self.P)

    @property
    def patches_local(self):
        return self.patches_local_.view(1, self.N*self.M, self.S_local, 3)
    
    @property
    def patches_local_monodisp(self):
        return self.patches_local_monodisp_.view(1, self.N*self.M, self.S_local, 1)

    @property
    def trajs_3d_world(self):
        return self.trajs_3d_world_.view(1, self.N*self.M, self.S_local, 3)
    

    @property
    def patches_local_weights(self):
        return self.patches_local_weights_.view(1, self.N*self.M, self.S_local, 1)

    @property
    def patches_local_vis(self):
        return self.patches_local_vis_.view(1, self.N*self.M, self.S_local, 1)
    
    @property
    def patches_local_static(self):
        return self.patches_local_static_.view(1, self.N*self.M, self.S_local, 1)
    
    @property
    def patches_monodisp(self):
        return self.patches_monodisp_.view(1, self.N*self.M, 1)
    
    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
    
    def init_motion(self):
        if self.n > 1:
            if self.cfg.slam.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.slam.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec
    
    def append_factors(self, ii, jj):
        """Add edges to factor graph 

        Args:
            ii (_type_): patch idx
            jj (_type_): frame idx
        """
        # project patch k from i to j
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        # temporal storage
        self.ii_new = self.ix[ii]
        self.jj_new = jj
        self.kk_new = ii

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.targets_3d = self.targets_3d[:,~m]
        self.weights = self.weights[:,~m]
        self.weights_pose = self.weights_pose[:,~m]

    def __image_gradient_2(self, images):
        images_pad = F.pad(images, (1,1,1,1), 'constant', 0)
        gray = images_pad.sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    
    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)
    
    def generate_patches(self, image):
        device = image.device
        B = 1
        # sample center
        # uniform
        if self.cfg.slam.PATCH_GEN == 'uniform':
            M_ = np.sqrt(self.M).round().astype(np.int32)
            grid_y, grid_x = utils.basic.meshgrid2d(B, M_, M_, stack=False, norm=False, device='cuda')
            grid_y = 8 + grid_y.reshape(B, -1)/float(M_-1) * (self.ht-16)
            grid_x = 8 + grid_x.reshape(B, -1)/float(M_-1) * (self.wd-16)
            coords = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2

        elif self.cfg.slam.PATCH_GEN == 'random':
            x = torch.randint(1, self.wd-1, size=[1, self.M], device="cuda")
            y = torch.randint(1, self.ht-1, size=[1, self.M], device="cuda")
            coords = torch.stack([x, y], dim=-1).float()
        
        elif self.cfg.slam.PATCH_GEN == 'sift':
            margin = 16
            image_array = self.local_window[-1].permute(1, 2, 0).detach().cpu().numpy() # H, W, C
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            gray = ((gray - gray.min()) * (255.0 / (gray.max() - gray.min()))).astype(np.uint8)
            sift = cv2.SIFT_create()
            
            kps = sift.detect(gray,None)
            kps, des = sift.compute(gray, kps)                
            
            # Iterate over each keypoint
            kps_pt = [x.pt for x in kps]
            kp_np = np.array(kps_pt)
            if len(kps_pt) > 0:
                mask = (kp_np[...,0] > margin) & (kp_np[...,0] < self.wd - margin) & (kp_np[...,1] > margin) & (kp_np[...,1] < self.ht - margin) 
                kp_np = kp_np[mask]
                np.random.shuffle(kp_np)
                kp_np = kp_np[:self.M]

            if len(kp_np) == self.M:
                xys = torch.from_numpy(kp_np).to(device)
            else:
                # if detector is smaller than num_anchors, add random points
                diff = self.M - len(kp_np)
                x = torch.randint(margin, self.wd-margin, size=[diff])
                y = torch.randint(margin, self.wd-margin, size=[diff])
                xy = torch.stack([x, y], dim=-1).float().to(device)
                kp_np = torch.concat([xy, torch.from_numpy(kp_np).to(device)], dim=0)
                xys = kp_np

            coords = xys[None,...].float()    # self.M, 2

            
        elif 'grid_grad' in self.cfg.slam.PATCH_GEN:
            rel_margin = 0.15
            num_expand = 8
            
            grid_size = int(self.cfg.slam.PATCH_GEN.split('_')[-1])
            num_grid = grid_size * grid_size
            grid_M = self.M // num_grid
            H_grid, W_grid = self.ht//grid_size, self.wd // grid_size
            
            g = self.__image_gradient_2(self.local_window[-1][None, None, ...])
      
            x = torch.rand((num_grid, num_expand * grid_M), device="cuda") * (1 - 2 * rel_margin) + rel_margin
            y = torch.rand((num_grid, num_expand * grid_M), device="cuda") * (1 - 2 * rel_margin) + rel_margin
            # map to coordinate
            offset = torch.linspace(0, grid_size-1, grid_size)
            offset_y, offset_x = torch.meshgrid(offset, offset)
            offset = torch.stack([offset_x, offset_y], dim=-1).to('cuda')
            offset = offset.view(-1,2)
            offset[...,0] = offset[...,0] * W_grid
            offset[...,1] = offset[...,1] * H_grid
        
            x_global = x.view(1, num_grid, -1) *W_grid  +  offset[...,0].view(1,-1,1) 
            y_global = y.view(1, num_grid, -1) *H_grid  +  offset[...,1].view(1,-1,1) 
        
            coords = torch.stack([x_global, y_global], dim=-1).float()    ## [1, N, 2]
            coords = rearrange(coords, 'b g n c -> b (g n) c')
            coords = torch.round(coords).unsqueeze(1)
            coords_norm = coords
            coords_norm[...,0] = coords_norm[...,0] / (self.wd - 1) * 2.0 - 1.0
            coords_norm[...,1] = coords_norm[...,0] / (self.ht - 1) * 2.0 - 1.0

            gg = F.grid_sample(g, coords_norm, mode='bilinear', align_corners=True)
            gg = gg[:,0,0]
            gg = rearrange(gg,'b (ng n) -> b ng n', ng=num_grid)
            ix = torch.argsort(gg, dim=-1)         
            x_global = torch.gather(x_global, 2, ix[:, :, -grid_M:])
            y_global = torch.gather(y_global, 2, ix[:, :, -grid_M:])
            coords = torch.concat([x_global, y_global], dim=-1).float()

        disps = torch.ones(B, 1, self.ht, self.wd, device="cuda")
        grid, _ = coords_grid_with_index(disps, device=self.poses_.device) 
        patches = altcorr.patchify(grid[0], coords, self.P//2).view(B, -1, 3, self.P, self.P)  # B, N, 3, p, p

        clr = altcorr.patchify(image.unsqueeze(0).float(), (coords + 0.5), 0).view(B, -1, 3)

        return patches, clr
    
    def map_point_filtering(self):
        coords = self.reproject()[...,self.P//2, self.P//2]
        ate = torch.norm(coords - self.targets_3d[...,:2],dim=-1)
        reproj_mask = (ate < self.cfg.slam.MAP_FILTERING_TH)
        self.weights[~reproj_mask] = 0
        self.weights_pose[~reproj_mask] = 0

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()
    
    def load(self):
        strict = True
        if self.cfg.model.init_dir != "":

            state_dict = torch.load(self.cfg.model.init_dir, map_location='cuda:0')
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.network.load_state_dict(state_dict, strict=strict)

    def load_weights(self):
        if self.cfg.model.mode == 'md_tracker':
            self.network = MDTracker(cfg=self.cfg).cuda()
        else:
            raise NotImplementedError
        self.load()
        self.network.eval()

    def align_depth(self, D_curr):
        if len(self.local_window_depth) == 0:
            return D_curr
        
        D_prev = self.local_window_depth[-1]
        mask = (D_prev > 0) & (D_curr > 0)
        D_prev_overlap = D_prev[mask]
        D_curr_overlap = D_curr[mask]

        if len(self.local_window_depth) == 1:
            median_prev = torch.median(D_prev_overlap)
        else:
            D_past_prev = self.local_window_depth[-2]
            mask_past = (D_past_prev > 0) & (D_prev > 0)
            D_past_prev_overlap = D_past_prev[mask_past]
            D_prev_overlap_combined = torch.cat([D_past_prev_overlap, D_prev_overlap])
            median_prev = torch.median(D_prev_overlap_combined)
        
        median_curr = torch.median(D_curr_overlap)
        scale = median_prev / median_curr
        D_scaled = scale * D_curr
        return D_scaled

    def preprocess(self, image, depth, intrinsics):
        """ Load the image and store in the local window
        """      
        if len(self.local_window) >= self.S:
            self.local_window.pop(0)
            self.local_window_depth.pop(0)
        self.local_window.append(image)
        self.local_window_depth.append(depth)

        self.intrinsics_[self.n] = intrinsics
        
        torch.cuda.empty_cache()
    

    def __edges(self):
        """Edge between keyframe patches and the all local frames
        """
        r = self.cfg.slam.S_slam
        local_start_fid = max((self.n - r), 0)
        local_end_fid = max((self.n - 0), 0)
        idx = torch.arange(0, self.n * self.M, device="cuda").reshape(self.n, self.M)
        kf_idx = idx[local_start_fid:local_end_fid:self.kf_stride].reshape(-1)
        
        return flatmeshgrid(
            kf_idx,
            torch.arange(max(self.n-self.S_slam, 0), self.n, device="cuda"), indexing='ij')
    

    def get_gt_trajs(self, xys, xys_sid):
        """Compute the gt trajectories from ground truth depth and camera pose

        Args:
            xys (tensor): B, N, 2
            xys_sid (tensor): B, N
        Returns:
            xy_gt (tensor): B, S, N, 2
            valid (tensor): B, S, N, 2
        """
        B, N = xys.shape[:2]
        S = len(self.local_window_depth_g)
   
        depths = torch.stack(self.local_window_depth_g, dim=0).unsqueeze(0).to(xys.device)   # B, S, C, H, W
        cams_c2w = torch.stack(self.local_window_cam_g, dim=0).unsqueeze(0).to(xys.device)   # B, S, C, H, W
        intrinsics = self.intrinsics[:,self.n-S:self.n].to(xys.device)
        
        assert len(self.local_window_cam_g) == len(self.local_window_depth_g)

        # back-project xy from each frame
        P0 = torch.empty(B, N, 4).to(xys.device)
        xy_depth = torch.empty(B, N, 1).to(xys.device)
        for s in range(S):
            mask = (xys_sid == s)
            xys_s = xys[mask].reshape(B, self.M, 2)
            depth_s = altcorr.patchify(depths[:,[s]].float(), xys_s, 0).reshape(B, self.M, 1)
            xy_depth[mask] = depth_s.reshape(-1, 1)
            P0[mask] = pops.back_proj(xys_s, depth_s, intrinsics[:,s], cams_c2w[:,s]).reshape(-1, 4)

        # project to all frame in the local window
        cams_w2c = torch.inverse(cams_c2w)
        xy_gt = pops.proj_to_frames(P0, intrinsics, cams_w2c)
        
        xy_gt = xy_gt[:,:S]
            
        # Detect NAN value
        xy_repeat = repeat(xys, 'b n c -> b s n c', s=S)
        invalid = torch.isnan(xy_gt) | torch.isinf(xy_gt)
        invalid_depth = (xy_depth <= 0) | torch.isnan(xy_depth) | torch.isinf(xy_depth)
        invalid_depth = repeat(invalid_depth, 'b n i -> b s n (i c)', s=S, c=2)
        invalid = invalid | invalid_depth
        xy_gt[invalid] = xy_repeat[invalid]
        valid = ~invalid

        return xy_gt, valid
    
    def get_queries(self):
        """return the query of the current local video window

        Returns:
            queries: (1, N, 3) in format (t, x, y)
        """

        S = len(self.local_window)
        xys = self.patches_[self.n-S:self.n, :, :2, self.P//2, self.P//2] 
        xys = xys.unsqueeze(0) # B, S, M, 2
        
        B = xys.shape[0]
         # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), 's -> b s m', b=B, m=self.M)
        

        xys = rearrange(xys[:, ::self.kf_stride], 'b s m c -> b (s m) c')
        xys_sid = rearrange(xys_sid[:,::self.kf_stride], 'b s m -> b (s m)')

        queries = torch.cat([xys_sid.unsqueeze(-1), xys], dim=2)

        return queries


    def get_patches_xy(self):
        S = len(self.local_window)
        # extract the patches from local windows 
        xys = self.patches_[self.n-S:self.n, :, :2, self.P//2, self.P//2]  # S, M, 2
        xys = xys.unsqueeze(0) # B, S, M, 2
        
        B = xys.shape[0]
         # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), 's -> b s m', b=B, m=self.M)
        xys = rearrange(xys, 'b s m c -> b (s m) c')
        xys_sid = rearrange(xys_sid, 'b s m -> b (s m)')
      
        coords_init = None
        if S > 1 and self.is_initialized :
            N = xys.shape[1]

            if self.cfg.slam.TRAJ_INIT == 'copy':
                coords_init = xys.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)
            
            elif self.cfg.slam.TRAJ_INIT == 'reproj':
                # init from reprojection
                ii = []
                jj = []
                kk = []
                for s in range(S-1):
                    patch_ii = torch.ones(self.M * (S-1)) * (self.n-S+s)
                    patch_jj = repeat(torch.arange(S-1) + self.n-S, 's -> (m s)', m=self.M)
                    patch_kk = repeat(torch.arange(self.M) + (self.n-S+s) * self.M, 'm -> (m s)', s=S-1)
                    ii.append(patch_ii)
                    jj.append(patch_jj)
                    kk.append(patch_kk)
                    
                ii = torch.cat(ii).long()
                jj = torch.cat(jj).long()
                kk = torch.cat(kk).long()
                coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
                
                coords = rearrange(coords, 'b (s2 m s1) p1 p2 c -> b s1 s2 m (p1 p2 c)', s1=S-1, s2=S-1, p1=1, p2=1)
                coords_init = rearrange(coords_init, 'b s1 (s2 m) c -> b s1 s2 m c', s2=S, m=self.M)
                patch_valids = repeat(self.patches_valid_[self.n-S: self.n-1], 's2 m -> b s1 s2 m c', b=B, s1=S-1, c=2).bool()
                coords_init[:,:S-1,:S-1][patch_valids] = coords[patch_valids]
                coords_init = rearrange(coords_init, 'b s1 s2 m c -> b s1 (s2 m) c')
 
        return xys, xys_sid, coords_init


    def _compute_sparse_tracks(
        self,
        rgbds,
        queries,
    ):
        B, T, C, H, W = rgbds.shape
        
        assert B == 1 and C == 4 # rgbd 
        rgbds = rgbds.reshape(B * T, C, H, W).float()
        rgbds = F.interpolate(rgbds, tuple(self.interp_shape), mode="bilinear")     # self.interp_shape = (384, 512)
        rgbds = rgbds.reshape(B, T, 4, self.interp_shape[0], self.interp_shape[1])

        video = rgbds[:,:,:3]

        queries = queries.clone()
        B, N, D = queries.shape
        assert D == 4
        # scale query position according to interp_shape
        queries[:, :, 1] *= self.interp_shape[1] / W
        queries[:, :, 2] *= self.interp_shape[0] / H


        if self.cfg.model.mode in ['md_tracker']:
            stats = {}
            tracks, _, depths, traj_static_3d_e, visibilities, dynamic_e,  __ = self.network(
                    rgbds=rgbds,
                    queries=queries,
                    iters=self.cfg.model.I,
                )
            if 'use_static_mask' in self.cfg.model and self.cfg.model.use_static_mask:
                tracks_static = traj_static_3d_e[...,:2]
                depths_static = traj_static_3d_e[...,2:]
                dyn_mask = dynamic_e > ( 1 - self.cfg.slam.STATIC_THRESHOLD)
                tracks[dyn_mask] = tracks_static[dyn_mask]
                depths[dyn_mask] = depths_static[dyn_mask]
            if 'use_static' in self.cfg.model and self.cfg.model.use_static:
                tracks = traj_static_3d_e[...,:2]
                depths = traj_static_3d_e[...,2:]

            stats['dynamic_e'] = dynamic_e

            if self.cfg.slam.backward_tracking and self.S_slam > self.cfg.model.S:
                tracks, depths, visibilities, stats = self._compute_backward_tracks(
                    rgbds, queries, tracks, depths, visibilities, stats
                )
        
        for i in range(len(queries)):
            queries_t = queries[i, :tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:3]
            
            visibilities[i, queries_t, arange] = 1.0

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
    
        return tracks, depths, visibilities, stats


    def _compute_backward_tracks(self, rgbds, queries, tracks, depths, visibilities, stats):
        inv_rgbds = rgbds.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_rgbds.shape[1] - inv_queries[:, :, 0] - 1

        inv_video = inv_rgbds[:,:,:3]

        inv_stats = {}


        if self.cfg.model.mode in ['md_tracker']:
            inv_traj_e, _, inv_depth_e, inv_traj_static_3d_e, inv_vis_e, inv_dynamic_e, _ = self.network(rgbds=inv_rgbds, queries=inv_queries, iters=self.cfg.model.I)

            if 'use_static_mask' in self.cfg.model and self.cfg.model.use_static_mask:
                inv_tracks_static = inv_traj_static_3d_e[...,:2]
                inv_depth_static = inv_traj_static_3d_e[...,2:]
                dyn_mask = inv_dynamic_e > ( 1 - self.cfg.slam.STATIC_THRESHOLD)
                inv_traj_e[dyn_mask] = inv_tracks_static[dyn_mask]
                inv_depth_e[dyn_mask] = inv_depth_static[dyn_mask]
            if 'use_static' in self.cfg.model and self.cfg.model.use_static:
                inv_traj_e = inv_traj_static_3d_e[...,:2]
                inv_depth_e = inv_traj_static_3d_e[...,2:]

            inv_stats['dynamic_e'] = inv_dynamic_e

            inv_tracks = inv_traj_e.flip(1)
            inv_depth = inv_depth_e.flip(1)
            inv_visibilities = inv_vis_e.flip(1)

            mask = tracks == 0

            tracks[mask] = inv_tracks[mask]
            depths[mask[...,[0]]] = inv_depth[mask[...,[0]]]
            visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]

            for key, value in stats.items():
                if key in ['dynamic_e']:
                    stats[key][mask[:, :, :, 0]] = inv_stats[key][mask[:, :, :, 0]]

        return tracks, depths, visibilities, stats


    def update_local(self, target_3d, weights, vis_e, static_e): 
        """
        target_3d: b (s1 m s) c
        weights: b (s1 m s) c
        vis_e: b (s1 m s)
        static_e: b (s1 m s)

        """
        # new edges id        
        ii = self.ii_new
        jj = self.jj_new
        kk = self.kk_new

        # put new target_3d into self.patches_local
        local_t = jj - ii
        local_id = local_t + (self.S_local + 1) // 2 - 1
        valid_mask = torch.logical_and(local_id >= 0, local_id < self.S_local)

        kk_valid = kk[valid_mask]
        local_id_valid = local_id[valid_mask]
        target_3d_valid = target_3d[:, valid_mask]
        weights_valid = weights[:, valid_mask]

        vis_e_valid = vis_e[:, valid_mask]
        static_e_valid = static_e[:, valid_mask]
        self.patches_local[:, kk_valid, local_id_valid, :3] = target_3d_valid
        self.patches_local_monodisp[:, kk_valid, local_id_valid] = target_3d_valid[...,2:]
        
        self.patches_local_vis[:, kk_valid, local_id_valid] = vis_e_valid[...,None].float()
        self.patches_local_static[:, kk_valid, local_id_valid] = static_e_valid[...,None].float()
        
        self.patches_local_weights[:, kk_valid, local_id_valid] = weights_valid[...,[0]]

        # print("update local")

    def get_window_trajs(self, only_coords=False):
        rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0)   # B, S, C, H, W
        dmaps = torch.stack(self.local_window_depth, dim=0).unsqueeze(0)   # B, S, C, H, W

        rgbds = torch.cat([rgbs, dmaps], dim=2)
        
        B, S_local, _, H, W = rgbs.shape

        queries = self.get_queries()
        
        depth_interp=[]
        for i in range(queries.shape[1]):
            depth_interp_i = bilinear_sample2d(dmaps[0,queries[:, i:i+1, 0].long()], 
                                queries[:, i:i+1, 1], queries[:, i:i+1, 2])
            depth_interp.append(depth_interp_i)

        depth_interp = torch.cat(depth_interp, dim=1)
        queries = smart_cat(queries, depth_interp, dim=-1)

        # put the queries into self.monodisp
        queries_disp = 1.0 / depth_interp.clamp(min=1e-2).reshape(1, -1, self.M, 1)
        queries_t_global = torch.arange(self.n-len(self.local_window), self.n)[::self.kf_stride]
        self.patches_monodisp_[queries_t_global] = queries_disp

        # pad repeated frames to make local window = S
        if rgbds.shape[1] < self.S_slam:
            repeat_rgbds = repeat(rgbds[:,-1], 'b c h w -> b s c h w', s=self.S-S_local)
            rgbds = torch.cat([rgbds, repeat_rgbds], dim=1)
        
        static_label = None
        coords_vars = None
        conf_label = None

        traj_e, depth_e, vis_e, stats = self._compute_sparse_tracks(rgbds=rgbds, queries=queries)
        
        # update local target, depth, vis_label
        # self.update_local(queries, traj_e, depth_e, S_local)
        local_target = traj_e
        local_depth = depth_e

        if 'VIS_THRESHOLD' in self.cfg.slam:
            vis_label = (vis_e > self.cfg.slam.VIS_THRESHOLD)   # B, S, N
        else:
            vis_label = (torch.ones_like(vis_e) > 0)

        padding = 20
        boundary_mask = (traj_e[...,0] >= padding) & (traj_e[...,0] < self.wd - padding) & (traj_e[...,1] >= padding) & (traj_e[...,1] < self.ht - padding) 

        vis_label_raw = (vis_label & boundary_mask).detach().clone()
        
        if 'dynamic_e' in stats and 'STATIC_THRESHOLD' in self.cfg.slam:
            static_e = 1 - stats['dynamic_e']
            static_th = torch.quantile(static_e,  (1 - self.cfg.slam.STATIC_QUANTILE))
            static_th = min(static_th.item(), self.cfg.slam.STATIC_THRESHOLD)
            static_label = static_e >= static_th

            # if not self.cfg.slam.use_static_all:
            #     vis_label = vis_label & static_label

        if 'var_e' in stats and 'CONF_THRESHOLD' in self.cfg.slam:
            coords_vars = torch.sqrt(stats['var_e'])
            conf_th = torch.quantile(coords_vars,  self.cfg.slam.CONF_QUANTILE, dim=2, keepdim=True)
            conf_th[conf_th < self.cfg.slam.CONF_THRESHOLD] = self.cfg.slam.CONF_THRESHOLD
            conf_label = coords_vars < conf_th
            vis_label = vis_label & conf_label

        local_target = local_target[:,:S_local]
        local_depth = local_depth[:, :S_local]
        vis_label = vis_label[:,:S_local]

        # update patches valid
        if self.is_initialized:
            query_valid = self.patches_valid_[self.n-len(self.local_window):self.n:self.kf_stride]
            valid_from_filter = (vis_label.sum(dim=1) > 3) 
            query_valid = torch.logical_or(query_valid.reshape(1,-1), valid_from_filter)
            self.patches_valid_[self.n-len(self.local_window):self.n:self.kf_stride] = query_valid.reshape(-1, self.M)

        stats = {
            'vis_e_raw': vis_label_raw[:,:S_local],
            'vis_label': None,
            'static_label': None,
            'conf_label': None,
            'coords_vars': None
        }

        if vis_label is not None: stats['vis_label'] = vis_label[:,:S_local]
        if static_label is not None: stats['static_label'] = static_label[:,:S_local]
        if conf_label is not None: stats['conf_label'] = conf_label[:,:S_local]
        if coords_vars is not None: stats['coords_vars'] = coords_vars[:,:S_local]
        
        return local_target, local_depth, vis_label, queries, stats
    
        
    def predict_target(self):
        with torch.no_grad():
            trajs, depths, vis_label, queries, stats, = self.get_window_trajs()

        B, S, N, C = trajs.shape
        disp = 1.0 / depths.clamp(min=1e-2)
        trajs_3d = torch.cat([trajs, disp], dim=-1)
        local_target_3d = rearrange(trajs_3d, 'b s n c -> b (n s) c')
        
        local_weight = torch.ones_like(local_target_3d[...,:2])

        vis_label = rearrange(vis_label, 'b s n -> b (n s)')
        local_weight[~vis_label] = 0

        padding = 20
        boundary_mask = (local_target_3d[...,0] >= padding) & (local_target_3d[...,0] < self.wd - padding) & (local_target_3d[...,1] >= padding) & (local_target_3d[...,1] < self.ht - padding) 
        local_weight[~boundary_mask] = 0 
        
        # check track length
        if self.n >= self.cfg.slam.MIN_TRACK_LEN:
            patch_valid = (local_weight > 0).any(dim=-1)
            patch_valid = rearrange(patch_valid, 'b (n s) -> b s n', s=S, n=N)
            patch_valid = (patch_valid.sum(dim=1) >= self.cfg.slam.MIN_TRACK_LEN)
            self.patches_valid_[self.n-S:self.n:self.kf_stride] = patch_valid.reshape(-1, self.M)

            track_len_mask = repeat(patch_valid, 'b n -> b (n s)', s=S)
            local_weight[~track_len_mask] = 0


        static_label = stats['static_label']
        static_label = rearrange(static_label, 'b s n -> b (n s)')
        local_weight_pose = local_weight.clone().detach()
        local_weight_pose[~static_label] = 0


        # append to global targets, weights
        self.targets_3d = torch.cat([self.targets_3d, local_target_3d], dim=1)
        self.weights = torch.cat([self.weights, local_weight], dim=1)
        self.weights_pose = torch.cat([self.weights_pose, local_weight_pose], dim=1)

        local_target_ = rearrange(local_target_3d, 'b (s1 m s) c -> b s s1 m c', s=S, m=self.M)
        local_weight_ = rearrange(local_weight, 'b (s1 m s) c -> b s s1 m c', s=S, m=self.M)
     
        local_vis = rearrange(stats['vis_e_raw'], 'b s n -> b (n s)')
        local_static = rearrange(stats['static_label'], 'b s n -> b (n s)')
        self.update_local(local_target_3d, local_weight, local_vis, local_static)

        vis_data = {
            'fid': self.n,
            'targets': local_target_[...,:2],
            'weights': local_weight_,
            'queries': queries
        }
        for key, value in stats.items():
            if value is not None:
                vis_data[key] = value


        self.visualizer.add_track(vis_data)


    def update_point_cloud(self):
        # static point cloud
        static_patches = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        static_points = (static_patches[...,self.P//2,self.P//2,:3] / static_patches[...,self.P//2,self.P//2,3:]).reshape(-1, 3)

        # dynamic point cloud
        patches_dyn = self.patches_local[:, :self.m]
        patches_dyn = rearrange(patches_dyn, 'b m s_local c -> b (m s_local) c')
        patches_dyn = patches_dyn[..., None, None]


        mid_id = (self.S_local + 1) // 2 - 1
        jj = self.ix[:self.m, None] + torch.arange(self.S_local, device=self.ix.device)[None, :] - mid_id
        jj = torch.clamp(jj, 0, self.N - 1)
        jj = jj.reshape(-1)

        dyn_points = pops.point_cloud(SE3(self.poses), patches_dyn, self.intrinsics, jj)
        dyn_points = dyn_points.reshape(self.m, self.S_local, 4)
        dyn_points = (dyn_points[...,:3]) / dyn_points[...,3:]

        valid_mask = self.patches_local_weights[:, :self.m]
        valid_mask_traj = (valid_mask.sum(dim=2) > 0).reshape(-1) # n_points

        dyn_points[valid_mask_traj] = static_points[valid_mask_traj][:,None]
        self.trajs_3d_world[:, :self.m] = dyn_points[None].clone()

        dyn_patches = self.patches_local[:, :self.m].detach().clone()
        static_patches_src = repeat(static_patches.detach().clone(), 'b m p1 p2 c -> b (m s_local) p1 p2 c', s_local=self.S_local)
        static_patches_world = SE3(self.poses)[:,jj, None, None] * static_patches_src
        static_patches_trg = pops.proj(static_patches_world, self.intrinsics[:,jj].detach(), depth=True)
        static_patches_trg = rearrange(static_patches_trg[:,:,self.P//2,self.P//2], 'b (m s_local) c -> b m s_local c', s_local=self.S_local)
        
        dyn_patches[:, valid_mask_traj] = static_patches_trg[:, valid_mask_traj]
        self.patches_local[:, :self.m] = dyn_patches

    def update(self):
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.n - self.cfg.slam.OPTIMIZATION_WINDOW if self.is_initialized else 1
        t0 = max(t0, 1)

        ep = 10
        lmbda = 1e-4
        bounds = [0, 0, self.wd, self.ht]
        Gs = SE3(self.poses)
        patches = self.patches
        patches_monodisp = self.patches_local[:, :, (self.S_local + 1) // 2 - 1, 2:]
        patches_local = self.patches_local

        for _ in range(self.cfg.slam.ITER):
            if self.cfg.slam.BA_mode == 'rgbd_dual_ba':
                Gs, patches = BA_rgbd_droid(Gs, patches, patches_monodisp, self.intrinsics.detach(), self.targets_3d[...,:2].detach(), self.targets_3d[...,2:].detach(), self.weights_pose.detach(), lmbda, self.ii, self.jj, self.kk, 
                    bounds, ep=ep, fixedp=t0, structure_only=False, loss=self.cfg.slam.LOSS, alpha=0.05)

                Gs, patches = BA_rgbd_droid(Gs, patches, patches_monodisp, self.intrinsics.detach(), self.targets_3d[...,:2].detach(), self.targets_3d[...,2:].detach(), self.weights.detach(), lmbda, self.ii, self.jj, self.kk, 
                    bounds, ep=ep, fixedp=t0, structure_only=True, loss=self.cfg.slam.LOSS, alpha=0.05)


            else:
                raise NotImplementedError
                
        
        self.patches_[:] = patches.reshape(self.N, self.M, 3, self.P, self.P)
        self.poses_[:] = Gs.vec().reshape(self.N, 7)
        self.patches_local_[:] = patches_local.reshape(self.N, self.M, self.S_local, 3)

        # 3D points culling
        if self.cfg.slam.USE_MAP_FILTERING:
            with torch.no_grad():
                self.map_point_filtering()
        
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,self.P//2,self.P//2,:3] / points[...,self.P//2,self.P//2,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

        self.update_point_cloud()

        
    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.detach().cpu().numpy()
        
        poses = poses[:,[0,1,2,6,3,4,5]]  # tx ty tz qx qy qz qw -> tx ty tz qw qx qy qz
        
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps
    
    def init_depth(self, patches, depth, mode='default'):
        
        if mode == 'default':
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s
        elif mode == 'dmap':
            # sample
            patches_depth = bilinear_sample2d(
                depth[None], 
                patches[:,:,0, 0, 0], 
                patches[:,:,1, 0, 0]
            )
            patches_disp = 1.0 / patches_depth.clamp(min=1e-2)
            patches[:,:,2] = patches_disp[:,0,:,None,None]

        return patches

    
    def __call__(self, tstamp, image, depth, intrinsics):
        """main function of tracking

        Args:
            tstamp (_type_): _description_
            image (_type_): 3, H, W
            depth (_type_): 1, H, W
            intrinsics (_type_): fx, fy, cx, cy

        Raises:
            Exception: _description_
        """
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)
        if self.visualizer is not None:
            self.visualizer.add_frame(image)

        # image preprocessing   
        self.preprocess(image, depth, intrinsics)
        
        # generate patches
        patches, clr = self.generate_patches(image)
        
        # depth initialization
        patches = self.init_depth(patches, depth, mode='dmap')
        
        self.patches_[self.n] = patches   

        if self.n % self.kf_stride == 0 and not self.is_initialized:  
            self.patches_valid_[self.n] = 1

        # pose initialization with motion model
        self.init_motion()
                
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter

        
        clr = clr[0]
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n] = self.n
            
        self.index_map_[self.n] = self.m
        
        self.counter += 1    

        self.n += 1
        self.m += self.M

        if (self.n - 1) % self.kf_stride == 0:  
            self.append_factors(*self.__edges())
            self.predict_target()

        if self.n == self.cfg.slam.num_init + 1 and not self.is_initialized:
            self.is_initialized = True            
            # one initialized, run global BA
            for itr in range(12):
                self.update()
            
            print("\n======================= \n [initialized] \n=======================\n")

        elif self.is_initialized:
            self.update()
            if (self.n - 1) % self.kf_stride == 0 and self.use_keyframe:  
                self.keyframe()
            else:
                self.keyframe_simple()

        torch.cuda.empty_cache()

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe_simple(self):
        """Keyframe selection
        """
        to_remove = self.ix[self.kk] < self.n - self.cfg.slam.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def keyframe(self):
        """Keyframe selection
        """
        k = self.n - self.cfg.slam.KEYFRAME_INDEX 
        if k %  self.kf_stride != 0:
            return

        i = self.n - self.cfg.slam.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.slam.KEYFRAME_INDEX + 1
        m = self.motionmag(i, k) + self.motionmag(j, k)
 
        if m / 2 < self.cfg.slam.KEYFRAME_THRESH:
            k = self.n - self.cfg.slam.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]
                self.patches_local_[i] = self.patches_local_[i+1]
                self.patches_local_vis_[i] = self.patches_local_vis_[i+1]
                self.patches_local_static_[i] = self.patches_local_static_[i+1]
                self.patches_local_weights_[i] = self.patches_local_weights_[i+1]
                self.patches_valid_[i] = self.patches_valid_[i+1]
                self.trajs_3d_world_[i] = self.trajs_3d_world_[i+1]

            self.n -= 1
            self.m-= self.M

            # remove keyframe from the local window
            self.local_window.pop(- self.cfg.slam.KEYFRAME_INDEX)
            self.local_window_depth.pop(- self.cfg.slam.KEYFRAME_INDEX)
            
        to_remove = self.ix[self.kk] < self.n - self.cfg.slam.REMOVAL_WINDOW
        self.remove_factors(to_remove)


    def depth_optim(self):
        """Given the camera pose, optimize the depth loss
        """

    def get_results(self, rgbs=None, dmaps=None, dmaps_gt=None, save_path=None):
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().matrix().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)


        pts_valid = self.patches_valid_[:self.counter].detach().cpu().numpy()

        intrinsics = self.intrinsics_[:self.counter].detach().cpu().numpy()

        grid_query_frames = np.arange(self.counter)[pts_valid.sum(axis=1) > 0]

        trajs_valid = self.patches_local_weights_[:self.counter,...,0]
        trajs_valid = (trajs_valid.sum(axis=2) > 0).detach().cpu().numpy()

        trajs_2d_disp = self.patches_local_[:self.counter].detach().cpu().numpy()

        trajs_static = self.patches_local_static_[:self.counter,...,0].detach().cpu().numpy()
        trajs_vis = self.patches_local_vis_[:self.counter,...,0].detach().cpu().numpy()
        

        if dmaps is not None:
            dmaps = np.array(dmaps, dtype=float)
        if rgbs is not None:
            rgbs = np.array(rgbs, dtype=float)
        if dmaps_gt is not None:
            dmaps_gt = np.array(dmaps_gt, dtype=float)

        results = {
            'cams_T_world': poses,  
            'intrinsics': intrinsics,
            'tstamps': tstamps,
            'trajs_2d_disp': trajs_2d_disp,
            'trajs_valid':trajs_valid,
            'trajs_static': trajs_static,
            'trajs_vis': trajs_vis,
            'grid_query_frames': grid_query_frames,
            'dmaps': dmaps, 
            'rgbs': rgbs,
            'dmaps_gt': dmaps_gt,
        }
        

        # save a pickle file
        import pickle
        if save_path is not None:
            with open(save_path, 'wb+') as f:
                pickle.dump(results, f)
            print(f"results saved to {save_path}")
      
        return results



