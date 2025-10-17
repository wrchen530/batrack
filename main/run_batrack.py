import os

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from main.batrack import BATRACK
from main.stream import sintel_rgbd_stream, tartanair_rgbd_stream, dataset_rgbd_stream, davis_stream
from main.utils import plot_trajectory, save_trajectory_tum_format, eval_metrics, load_traj

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig):

    slam = None
    skip = 0

    imagedir, calib, stride, skip, end = cfg.data.imagedir, cfg.data.calib, cfg.data.stride, cfg.data.skip, cfg.data.end
    depthdir = cfg.data.depthdir
    depthdir_gt = cfg.data.depthdir_gt if 'depthdir_gt' in cfg.data else depthdir

    # use gt intrinsics or not
    input_intrinsics = cfg.data.input_intrinsics if 'input_intrinsics' in cfg.data else False

    logger.info("Configuration - depthdir: %s, depthdir_gt: %s, input_intrinsics: %s", 
                depthdir, depthdir_gt, input_intrinsics)

    if os.path.isdir(imagedir):
        if cfg.data.traj_format == 'sintel':
            dataloader = sintel_rgbd_stream(imagedir, depthdir, depthdir_gt, calib, stride, skip, end, input_intrinsics)
        elif cfg.data.traj_format == 'tartanair':
            dataloader = tartanair_rgbd_stream(imagedir, depthdir, depthdir_gt, calib, stride, skip, end)
        elif cfg.data.traj_format == 'davis':
            dataloader = davis_stream(imagedir, depthdir, calib, stride, skip, end)
        else:
            dataloader = dataset_rgbd_stream(imagedir, depthdir, calib, stride, skip, mode=cfg.data.traj_format)
        
    else:
        logger.error("Image directory not found: %s", imagedir)
        logger.error("Trajectory format: %s", cfg.data.traj_format)
        raise ValueError(f"Invalid image directory: {imagedir}")
       
    image_list = []
    depth_list = []
    depth_list_gt = []
    intrinsics_list = []
    start_time = time.time()
    for i, (t, image, depth, depth_gt, intrinsics) in enumerate(tqdm(dataloader)):
        if t == -1: break
    
        depth = depth.clip(1e-2, 1e2)
        
        image_list.append(image)
        depth_list.append(depth)
        depth_list_gt.append(depth_gt)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        depth = torch.from_numpy(depth).permute(2,0,1).cuda().float()
        intrinsics = torch.from_numpy(intrinsics).cuda()
        # initialization
        if slam is None:
            slam = BATRACK(cfg, ht=image.shape[1], wd=image.shape[2])

        slam(t, image, depth, intrinsics)

    end_time = time.time()
    total_time = end_time - start_time
    num_frames = len(image_list)
    
    logger.info("Processing completed:")
    logger.info("  Total time: %.2f seconds", total_time)
    logger.info("  FPS: %.2f", num_frames / total_time)
    logger.info("  Time per frame: %.4f seconds", total_time / num_frames)
    
    pred_traj = slam.terminate()

    if 'gt_traj' in cfg.data and cfg.data.gt_traj != '':
        gt_traj = load_traj(cfg.data.gt_traj, cfg.data.traj_format, skip=cfg.data.skip, stride=cfg.data.stride, end=cfg.data.end)
        
    else:
        gt_traj = None

    os.makedirs(f"{cfg.data.savedir}/{cfg.data.name}", exist_ok=True)

    pred_traj = list(pred_traj)

    if cfg.save_results:
        save_path = f"{cfg.data.savedir}/{cfg.data.name}/results.pkl"
        slam.get_results(rgbs=image_list, dmaps=depth_list, dmaps_gt=depth_list_gt, save_path=save_path)

    if cfg.save_trajectory:
        save_trajectory_tum_format(pred_traj, f"{cfg.data.savedir}/{cfg.data.name}/batrack_traj.txt")

    if cfg.save_video:
        slam.visualizer.save_video(filename=cfg.slam.PATCH_GEN)

    if cfg.save_plot:
        plot_trajectory(pred_traj, gt_traj=gt_traj, title=f"Trajectory Prediction for {cfg.exp_name}", filename=f"{cfg.data.savedir}/{cfg.data.name}/traj_plot.pdf")

    if gt_traj is not None:
        ate, rpe_trans, rpe_rot = eval_metrics(pred_traj, gt_traj=gt_traj, seq=cfg.exp_name, filename=os.path.join(cfg.data.savedir,cfg.data.name, 'eval_metrics.txt'))
        with open(os.path.join(cfg.data.savedir, 'error_sum.txt'), 'a+') as f:
            line = f"{cfg.data.name:<20} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
            f.write(line)
            line = f"{ate:.5f}\n{rpe_trans:.5f}\n{rpe_rot:.5f}\n"
            f.write(line)


if __name__ == '__main__':
    main()
