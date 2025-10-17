import os

import torch
import numpy as np


from model.refine_net import RefineNet
from model.trainer import global_alignment_loop
import argparse

def test_all(result_dir, grid_size=4, scenes=None, align_depth=True, loss_weight_dict={}, niter=200, fixed_pose=False, fixed_K=True):
    os.makedirs(result_dir, exist_ok=True)
    scale_mode = 'exp'
    if scenes is None:
        scenes = os.listdir(result_dir)
    print(f"Processing scenes: {scenes}")

    device = torch.device('cuda')

    for scene in scenes:
        result_path = os.path.join(result_dir, scene, 'results.pkl')
        if not os.path.exists(result_path):
            print(f"Skipping {scene}: {result_path} not found")
            continue

        print(f"\nProcessing scene: {scene}")
        refine_intrinsics = (not fixed_K)
        refine_net = RefineNet(device=device, result_path=result_path, scale_mode=scale_mode, grid_size=grid_size, align_depth=align_depth, loss_weight_dict=loss_weight_dict, refine_intrinsics=refine_intrinsics, verbose=False)


        refine_net.to(device)
        global_alignment_loop(refine_net, lr=1e-2, niter=niter, schedule='cosine', lr_min=1e-6, fixed_K=fixed_K, fixed_pose=fixed_pose)
        

        ba_results = refine_net.get_results()
        results_save_path = os.path.join(result_dir, scene, 'results_refined.pkl')
        
        import pickle

        with open(results_save_path, 'wb+') as f:
            pickle.dump(ba_results, f)
        print(f"Refined results saved to {results_save_path}")

    print("All refinements completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DAVIS dataset with global alignment')
    parser.add_argument('--result_dir', type=str, default='', help='Result directory path')
    parser.add_argument('--scenes', nargs='*', default=None, help='List of scenes to process. If not provided, processes all folders in result_dir')
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size')
    parser.add_argument('--niter', type=int, default=200, help='Number of iterations')
    parser.add_argument('--fixed_pose', action='store_true', default=False, help='Fix pose during optimization')
    parser.add_argument('--fixed_K', action='store_true', default=False, help='Fix intrinsics during optimization')

    args = parser.parse_args()

    result_dir = args.result_dir
    scenes = args.scenes

    loss_weight_dict = {
        'spatial_loss': 5.0,
        'inter_frame_loss': 0.3,
        'pts_3d_loss': 1.0,
        'cam_smooth_vec_loss': 1.0,
        'scale_smoothness_loss': 0.3,
    }

    test_all(
        result_dir=result_dir, 
        grid_size=args.grid_size, 
        scenes=scenes, 
        loss_weight_dict=loss_weight_dict, 
        niter=args.niter, 
        fixed_pose=args.fixed_pose,
        fixed_K=args.fixed_K
    )