import os
import datetime

import torch
import numpy as np

from model.refine_net import RefineNet
from model.trainer import global_alignment_loop
from model.utils import eval_depth
import argparse

def test_all(result_dir, grid_size=4, scenes=None, align_depth=True, loss_weight_dict={}, depth_min=1e-2, depth_max=1e2, niter=200, scaling='median', fixed_pose=False, fixed_K=False):
    os.makedirs(result_dir, exist_ok=True)
    scale_mode = 'exp'
    if scenes is None:
        scenes = os.listdir(result_dir)
    print(scenes)

    save_file = os.path.join(result_dir, f'eval_depth.txt')
    with open(save_file, 'a+') as f:
         f.write(f'{datetime.datetime.now()}\n')

    device = torch.device('cuda')

    abs_rel_list = []
    a1_list = []
    a2_list = []
    a3_list = []

    for scene in scenes:
        scene_name = str(scene)
        result_path = os.path.join(result_dir, scene, 'results.pkl')
        if not os.path.exists(result_path):
            print(f"error: {scene}", result_path)
            continue


        refine_intrinsics = (not fixed_K)

        refine_net = RefineNet(device=device, result_path=result_path, scale_mode=scale_mode, grid_size=grid_size, align_depth=align_depth, loss_weight_dict=loss_weight_dict, refine_intrinsics=refine_intrinsics, verbose=False)

        refine_net.to(device)
        global_alignment_loop(refine_net, lr=1e-2, niter=niter, schedule='cosine', lr_min=1e-6, fixed_K=fixed_K, fixed_pose=fixed_pose)

        results = eval_depth(refine_net, depth_min=depth_min, depth_max=depth_max, scaling=scaling, scene_name=scene_name)

        abs_rel_list.append(results['final'][0])
        a1_list.append(results['final'][5])
        a2_list.append(results['final'][6])
        a3_list.append(results['final'][7])

        with open(save_file, 'a+') as f:
            f.write(f'{scene_name}\n')
            for key, value in results.items():
                f.write(f'{key}: abs_rel, a1, a2, a3\n')
                f.write(f'{value[0]:.4f}\n{value[5]:.4f}\n{value[6]:.4f}\n{value[7]:.4f}\n')
            f.write('\n')

    avg_abs_rel = np.mean(abs_rel_list)
    avg_a1 = np.mean(a1_list)
    avg_a2 = np.mean(a2_list)
    avg_a3 = np.mean(a3_list)

    print(f'Average abs_rel (final): {avg_abs_rel:.4f}')
    print(f'Average a1 (final): {avg_a1:.4f}')
    print(f'Average a2 (final): {avg_a2:.4f}')
    print(f'Average a3 (final): {avg_a3:.4f}')


    with open(save_file, 'a+') as f:
        f.write(f'Average abs_rel (final): {avg_abs_rel:.4f}\n')
        f.write(f'Average a1 (final): {avg_a1:.4f}\n')
        f.write(f'Average a2 (final): {avg_a2:.4f}\n')
        f.write(f'Average a3 (final): {avg_a3:.4f}\n')


    print("save results to ", save_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DAVIS dataset with global alignment')
    parser.add_argument('--result_dir', type=str, default='', help='Result directory path')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size')
    parser.add_argument('--niter', type=int, default=200, help='Number of iterations')
    parser.add_argument('--fixed_pose', action='store_true', default=False, help='Fix pose during optimization')
    parser.add_argument('--fixed_K', action='store_true', default=False, help='Fix intrinsics during optimization')

    args = parser.parse_args()

    result_dir = args.result_dir


    scenes = [ 'Standing01', 'Standing02',  'RoadCrossing03', 'RoadCrossing04', 'RoadCrossing05', 'RoadCrossing06', 'RoadCrossing07_depth']

    loss_weight_dict = {
        'spatial_loss': 5.0,
        'inter_frame_loss': 0.3,
        'pts_3d_loss': 1.0,
        # 'cam_smooth_vec_loss': 1.0,
    }

    test_all(
        result_dir, grid_size=args.grid_size, scenes=scenes, vis=False, 
        loss_weight_dict=loss_weight_dict, niter=args.niter, 
        depth_max=1e2, depth_min=1e-2, scaling='median',
        fixed_pose=True, fixed_K=True
    )