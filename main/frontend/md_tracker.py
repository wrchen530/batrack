import torch
import torch.nn as nn
from einops import rearrange, repeat

from main.frontend.core.cotracker.blocks import (
    BasicEncoder,
    CorrBlock,
    UpdateFormer,
    MotionLabelBlock,
)
from main.frontend.core.model_utils import meshgrid2d, bilinear_sample2d, smart_cat
from main.frontend.core.embeddings import (
    get_3d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    Embedder_Fourier
)


torch.manual_seed(0)


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cpu"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


def sample_pos_embed(grid_size, embed_dim, coords):
    pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
    pos_embed = (
        torch.from_numpy(pos_embed)
        .reshape(grid_size[0], grid_size[1], embed_dim)
        .float()
        .unsqueeze(0)
        .to(coords.device)
    )
    sampled_pos_embed = bilinear_sample2d(
        pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1]
    )
    return sampled_pos_embed


class MDTracker(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(MDTracker, self).__init__()
        self.cfg = cfg.model

        # if getattr(cfg, "Embed3D", None) == None:
        if 'Embed3D' in self.cfg:
            self.Embed3D = self.cfg.Embed3D
        else:
            self.Embed3D = True

        self.use_log_depth = self.cfg.use_log_depth if 'use_log_depth' in self.cfg else False

        self.static_iters = self.cfg.static_iters
        self.S = self.cfg.sliding_window_len
        self.stride = cfg.model.model_stride
        self.dynamic_mask_detach = self.cfg.dynamic_mask_detach

        self.interp_shape = (384, 512)
        self.model_resolution = self.interp_shape
        self.hidden_dim = self.cfg.hidden_dim if 'hidden_dim' in self.cfg else 256
        self.latent_dim = self.cfg.latent_dim if 'latent_dim' in self.cfg else 128
        self.corr_levels = self.cfg.corr_levels if 'corr_levels' in self.cfg else 4
        self.corr_radius = self.cfg.corr_radius if 'corr_radius' in self.cfg else 3

        self.add_space_attn = self.cfg.add_space_attn  

        self.depth_drop_rate = self.cfg.depth_drop_rate if 'depth_drop_rate' in self.cfg else 0.0
        
        # fmap_input_dim = 4 if self.cfg.fmap_input_mode == 'rgbd' else 3   # default: True
        fmap_input_dim = 3
        self.input_dim = 456
        self.fnet = BasicEncoder(
            input_dim=fmap_input_dim, output_dim=self.latent_dim, norm_fn="instance", dropout=0, stride=self.stride
        )

        self.updateformer_type = 'updateformer' if 'updateformer_type' not in self.cfg else self.cfg.updateformer_type

        self.fix_track_mask = self.cfg.fix_track_mask if 'fix_track_mask' in self.cfg else False



        if self.updateformer_type == 'updateformer':
            self.updateformer = UpdateFormer(
                space_depth=self.cfg.space_depth,
                time_depth=self.cfg.time_depth,
                input_dim=self.input_dim,
                hidden_size=self.cfg.hidden_size,
                num_heads=self.cfg.num_heads,
                output_dim=self.latent_dim + 3,
                mlp_ratio=4.0,
                add_space_attn=self.cfg.add_space_attn,
            )    


            self.updateformer_dyn = UpdateFormer(
                space_depth=self.cfg.space_depth_dyn,
                time_depth=self.cfg.time_depth_dyn,
                input_dim=self.input_dim,
                hidden_size=self.cfg.hidden_size,
                num_heads=self.cfg.num_heads,
                output_dim=self.latent_dim + 3,
                mlp_ratio=4.0,
                add_space_attn=self.cfg.add_space_attn
            )
     
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        # predict dynamic track
        if 'motion_label_block' in self.cfg:
            self.motion_label_block = MotionLabelBlock(cfg=self.cfg, S=self.S)
        else:
            self.motion_label_block = None

        self.embed3d = Embedder_Fourier(
            input_dim=3, max_freq_log2=10.0, N_freqs=10, include_input=True
        )
        self.embedConv = nn.Conv2d(self.latent_dim+63,
                            self.latent_dim, 3, padding=1) 

        self.zeroMLPflow = nn.Linear(195, 130)


    def depth_process(self, depth_raw):
        if self.use_log_depth:
            depth = torch.log(depth_raw.clamp(min=1e-3))
        else:
            depth = depth_raw
        return depth 

    def depth_process_inv(self, depth):
        if self.use_log_depth:
            depth_raw = torch.exp(depth)
        else:
            depth_raw = depth
        
        return depth_raw
    
    def _get_fmaps(self, rgbs):
        """Extract Image Features
        Args:
            rgbs: [B*S, C, H, W]
        """
        fmaps_ = self.fnet(rgbs)    # [B*S, C, H / stride, W / stride]
        return fmaps_

    
    def forward_iteration(
        self,
        fmaps,
        dmaps,
        coords_init,
        coords_dyn_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
    ):
        
        B, S_init, N, D = coords_init.shape
        
        assert D == 3
        assert B == 1

        B, S, __, H8, W8 = fmaps.shape

        device = fmaps.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            coords_dyn = torch.cat(
                [coords_dyn_init, coords_dyn_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()
            coords_dyn = coords_dyn_init.clone()

        fcorr_fn = CorrBlock(
            fmaps, num_levels=self.corr_levels, radius=self.corr_radius
        )

        ffeats = feat_init.clone()
        ffeats_static = feat_init.clone()

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=self.input_dim,
            coords=coords,
        )
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)
        pos_embed_static = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=self.input_dim,
            coords=coords - coords_dyn,
        )
        pos_embed_static = rearrange(pos_embed_static, "b e n -> (b n) e").unsqueeze(1)

        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.input_dim, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )

        coord_predictions = []
        coord_depth_predictions = []
        coord_static_predictions = []

        for __ in range(iters):
            coords = coords.detach()    # B, S, N, 3
            fcorr_fn.corr(ffeats)

            fcorrs = fcorr_fn.sample(coords[...,:2])  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, 64, cat_coords=True)   # B*N, S, 195
            flows_cat =  self.zeroMLPflow(flows_cat)


            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )

            if self.fix_track_mask:
                concat = (
                    torch.cat([track_mask, vis_init], dim=-1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, S, 2)
                )
            else:
                concat = (
                    torch.cat([track_mask, vis_init], dim=2)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, S, 2)
                )


            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)
            # pdb.set_trace()

            x = transformer_input + pos_embed + times_embed

            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer(x)

            delta = rearrange(delta, " b n t d -> (b n) t d")


            delta_coords_ = delta[:, :, :3]

            delta_feats_ = delta[:, :, 3:]

            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)

            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_

            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C

            coords = coords + delta_coords_.reshape(B, N, S, 3).permute(0, 2, 1, 3)
            coords_out = coords.clone()
            coords_out[..., :2] *= float(self.stride)
            
            coords_out[..., 2] = coords_out[..., 2] /self.Dz
            coords_out[..., 2] = coords_out[..., 2]*(self.d_far-self.d_near) + self.d_near
            coords_out[..., 2] = self.depth_process_inv(coords_out[..., 2])

            coord_predictions.append(coords_out[..., :2])
            coord_depth_predictions.append(coords_out[...,2:])
        
        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(
            B, S, N
        )   
        if self.motion_label_block is not None:
            dynamic_e = self.motion_label_block(ffeats, coords).squeeze(2)
        else:
            dynamic_e = torch.ones(B, N).to(coords.device)

        # total flow -> dynamic flow
        coords_total = coords.clone().detach()
        dynamic_mask = dynamic_e.unsqueeze(1).unsqueeze(3).repeat(1, S, 1, 3)
        if self.dynamic_mask_detach:
            dynamic_mask = torch.sigmoid(dynamic_mask).detach()
        else:
            dynamic_mask = torch.sigmoid(dynamic_mask)

        static_iters = self.static_iters
        for __ in range(static_iters):
            coords_dyn = coords_dyn.detach()
            coords_static = (coords_total - coords_dyn).detach()
            fcorr_fn.corr(ffeats_static)

            fcorrs = fcorr_fn.sample(coords_static[...,:2])  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords_static - coords_static[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, 64, cat_coords=True)   # B*N, S, 195
            flows_cat =  self.zeroMLPflow(flows_cat)

            ffeats_static_ = ffeats_static.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )

            if self.fix_track_mask:
                concat = (
                    torch.cat([track_mask, vis_init], dim=-1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, S, 2)
                )
            else:
                concat = (
                    torch.cat([track_mask, vis_init], dim=2)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, S, 2)
                )

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_static_, concat], dim=2)

            x = transformer_input + pos_embed_static + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer_dyn(x)
            delta = rearrange(delta, " b n t d -> (b n) t d")

            delta_coords_dyn_ = delta[:, :, :3]
            delta_feats_ = delta[:, :, 3:]

            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_static_ = ffeats_static.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)

            ffeats_static_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_static_

            ffeats_static = ffeats_static_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C

            coords_dyn = coords_dyn + delta_coords_dyn_.reshape(B, N, S, 3).permute(0, 2, 1, 3) 

            coords_dyn_out = coords_dyn.clone()
            coords_static_out = (coords_total.detach() - coords_dyn_out * dynamic_mask)

            coords_static_out[..., :2] *= float(self.stride)
            
            coords_static_out[..., 2] = coords_static_out[..., 2]/self.Dz
            coords_static_out[..., 2] = coords_static_out[..., 2]*(self.d_far-self.d_near) + self.d_near
            coords_static_out[..., 2] = self.depth_process_inv(coords_static_out[..., 2])

            coord_static_predictions.append(coords_static_out)

        return coord_predictions, coord_depth_predictions, coord_static_predictions, vis_e, dynamic_e, feat_init
    

    def forward(self, rgbds, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbds.shape
        B, N, __ = queries.shape

        device = rgbds.device
        assert B == 1
        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consequtively
        
        #Step2: sort the points via their first appeared time
        first_positive_inds = queries[:, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]
        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )
        

        rgbds[:, :, :3, ...] = 2 * (rgbds[:, :, :3, ...] / 255.0) - 1.0
        depth_all = self.depth_process(rgbds[:, :, 3,...])

        if self.use_log_depth:
            d_near = self.d_near = depth_all.min().item()
            d_far = self.d_far = depth_all.max().item()
        else:
            d_near = self.d_near = depth_all[depth_all>0.01].min().item()
            d_far = self.d_far = depth_all[depth_all>0.01].max().item()
        self.Dz = Dz = W//self.stride


        traj_e = torch.zeros((B, T, N, 2), device=device)
        depth_e = torch.zeros((B, T, N, 1), device=device)
        traj_static_e = torch.zeros((B, T, N, 3), device=device)
        vis_e = torch.zeros((B, T, N), device=device)
        dynamic_e = torch.zeros((B, T, N), device=device)

         # filter those points never appear points during 1 - T
        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        
        # these are logits, so we initialize visibility with something that would give a value close to 1 after softmax
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        coords_init = queries[:, :, 1:].reshape(B, 1, N, 3).repeat(
            1, self.S, 1, 1
        ) 
        coords_init[..., :2] /= float(self.stride)

        coords_init[..., 2] = (
                self.depth_process(coords_init[..., 2]) - d_near
                )/(d_far-d_near)
        coords_init[..., 2] *= Dz

        #Step3: initial the regular grid   
        gridx = torch.linspace(0, W//self.stride - 1, W//self.stride)
        gridy = torch.linspace(0, H//self.stride - 1, H//self.stride)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        gridxy = torch.stack([gridx, gridy], dim=-1).to(rgbds.device).permute(
            2, 1, 0
        )

        ind = 0

        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        coords_dyn_init_ = torch.zeros_like(coords_init_)
        vis_init_ = vis_init[:, :, sort_inds].clone()

        prev_wind_idx = 0
        fmaps_ = None
        vis_predictions = []
        dynamic_predictions = []
        coord_predictions = []
        coord_depth_predictions = []
        # coord_prob_predictions = []

        coord_static_flow_predictions = []
        coord_static_depth_predictions = []

        wind_inds = []
        while ind < T - self.S // 2:
            rgbds_seq = rgbds[:, ind : ind + self.S]

            S = S_local = rgbds_seq.shape[1]
            if S < self.S:
                rgbds_seq = torch.cat(
                    [rgbds_seq, rgbds_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )

                S = rgbds_seq.shape[1]
                
            rgbs_ = rgbds_seq.reshape(B * S, C, H, W)[:,:3]
            depths_ = rgbds_seq.reshape(B * S, C, H, W)[:,3:].clone()

            depths_ = self.depth_process(depths_)
            depths_ = (depths_ - d_near)/(d_far-d_near)            
            depths_dn = nn.functional.interpolate(
                    depths_, scale_factor=1.0 / self.stride, mode="nearest")
            depths_dnG = depths_dn*Dz

            gridxyz = torch.cat([gridxy[None,...].repeat(
                                depths_dn.shape[0],1,1,1), depths_dnG], dim=1)
            

            if self.Embed3D:
                gridxyz_nm = gridxyz.clone()
                gridxyz_nm[:,0,...] = (gridxyz_nm[:,0,...]-gridxyz_nm[:,0,...].min())/(gridxyz_nm[:,0,...].max()-gridxyz_nm[:,0,...].min())
                gridxyz_nm[:,1,...] = (gridxyz_nm[:,1,...]-gridxyz_nm[:,1,...].min())/(gridxyz_nm[:,1,...].max()-gridxyz_nm[:,1,...].min())
                gridxyz_nm[:,2,...] = (gridxyz_nm[:,2,...]-gridxyz_nm[:,2,...].min())/(gridxyz_nm[:,2,...].max()-gridxyz_nm[:,2,...].min())
                gridxyz_nm = 2*(gridxyz_nm-0.5)
                _,_,h4,w4 = gridxyz_nm.shape
                gridxyz_nm = gridxyz_nm.permute(0,2,3,1).reshape(S*h4*w4, 3)
                featPE = self.embed3d(gridxyz_nm).view(S, h4, w4, -1).permute(0,3,1,2)

                if is_train and self.depth_drop_rate > 0.0:
                    mask = torch.rand(featPE.shape[0]) > self.depth_drop_rate
                    mask = mask.to(featPE.device)
                    featPE = featPE * mask.float().view(-1, 1, 1, 1)

                if fmaps_ is None:
                    fmaps_ = torch.cat([self.fnet(rgbs_),featPE], dim=1) 
                    fmaps_ = self.embedConv(fmaps_)
                else:
                    fmaps_new = torch.cat([self.fnet(rgbs_[self.S // 2 :]),featPE[self.S // 2 :]], dim=1) 
                    fmaps_new = self.embedConv(fmaps_new)
                    fmaps_ = torch.cat(
                        [fmaps_[self.S // 2 :], fmaps_new], dim=0
                    )
            else:        
                if fmaps_ is None:
                    fmaps_ = self.fnet(rgbs_)
                else:
                    fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                    )

            fmaps = fmaps_[:, :self.latent_dim].reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )
            dmaps = depths_dnG.reshape(B, S, 1, H // self.stride, W // self.stride)

            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1

            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]

                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)

                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)

            if prev_wind_idx > 0:
                new_coords_2d = coords[-1][:, self.S // 2 :].clone()
                new_coords_2d /= float(self.stride)

                new_coords_depth = coords_depth[-1][:, self.S // 2 :].clone()
                new_coords_depth = (self.depth_process(new_coords_depth)-d_near)/(d_far-d_near)
                new_coords_depth = new_coords_depth*Dz 

                new_coords = torch.cat([new_coords_2d, new_coords_depth], dim=-1)

                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2 :, :prev_wind_idx] = new_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)

                
                # dynamic
                new_coords_dyn_2d = new_coords_2d -  coords_static_3d[-1][:, self.S // 2 :, :, :2].clone()
                new_coords_dyn_2d /= float(self.stride)

                new_coords_dyn_depth = new_coords_depth - coords_static_3d[-1][:, self.S // 2 :, :, 2:].clone()
                new_coords_dyn_depth = (self.depth_process(new_coords_dyn_depth)-d_near)/(d_far-d_near)
                new_coords_dyn_depth = new_coords_dyn_depth*Dz

                new_coords_dyn = torch.cat([new_coords_dyn_2d, new_coords_dyn_depth], dim=-1)
                coords_dyn_init_[:, : self.S // 2, :prev_wind_idx] = new_coords_dyn
                coords_dyn_init_[:, self.S // 2 :, :prev_wind_idx] = new_coords_dyn[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)


                new_vis = vis[:, self.S // 2 :].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2 :, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            coords, coords_depth, coords_static_3d, vis, dynamic, __ = self.forward_iteration(
                fmaps=fmaps,    
                dmaps=dmaps,
                coords_init=coords_init_[:, :, :wind_idx],
                coords_dyn_init=coords_dyn_init_[:, :, :wind_idx],
                feat_init=feat_init[:, :, :wind_idx],
                vis_init=vis_init_[:, :, :wind_idx],
                track_mask=track_mask_[:, ind : ind + self.S, :wind_idx],
                iters=iters,
            )

            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                dynamic_predictions.append(torch.sigmoid(repeat(dynamic, 'b n -> b s n', s=S_local)))
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                coord_depth_predictions.append([coord_depth[:, :S_local] for coord_depth in coords_depth])
                coord_static_flow_predictions.append([coord_static[:, :S_local, :, :2] for coord_static in coords_static_3d])
                coord_static_depth_predictions.append([coord_static[:, :S_local, :, 2:] for coord_static in coords_static_3d])
                wind_inds.append(wind_idx)

            traj_e[:, ind : ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            depth_e[:, ind : ind + self.S, :wind_idx] = coords_depth[-1][:, :S_local]
            traj_static_e[:, ind : ind + self.S, :wind_idx] = coords_static_3d[-1][:, :S_local]
            vis_e[:, ind : ind + self.S, :wind_idx] = vis[:, :S_local]
            dynamic_e[:, ind : ind + self.S, :wind_idx] = repeat(dynamic, 'b n -> b s n', s=S_local)

            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2

            prev_wind_idx = wind_idx

        traj_e = traj_e[:, :, inv_sort_inds]
        depth_e = depth_e[:, :, inv_sort_inds]
        traj_static_e = traj_static_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        vis_e = torch.sigmoid(vis_e)

        dynamic_e = dynamic_e[:, :, inv_sort_inds]
        dynamic_e = torch.sigmoid(dynamic_e)

   
        train_data = (
            (vis_predictions, 
             coord_predictions,
             coord_depth_predictions, 
             coord_static_flow_predictions,
             coord_static_depth_predictions,
             dynamic_predictions, 
             wind_inds, 
             sort_inds
            )
            if is_train
            else None
        )
        return traj_e, feat_init, depth_e, traj_static_e, vis_e, dynamic_e, train_data


