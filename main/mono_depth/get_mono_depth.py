import numpy as np
import os
import cv2
import os
import glob
from tqdm import tqdm

import argparse

def intrinsics_to_fov(K, depth):
    fov_ = np.rad2deg(
        2
        * np.arctan(
            depth.shape[-1]
            / (2 * K[0, 0])
        )
    )
    return fov_


def align_depth(mono_depth_path, metric_depth_path, scene_name, datapath, save_depth_dir, save_K_dir):
    os.makedirs(save_depth_dir, exist_ok=True)
    os.makedirs(save_K_dir, exist_ok=True)

    print(datapath)

    image_list = sorted(glob.glob(os.path.join("%s" % (datapath), "*.jpg")))
    image_list += sorted(glob.glob(os.path.join("%s" % (datapath), "*.png")))

    mono_disp_paths = sorted(
        glob.glob(
            os.path.join("%s/%s" % (mono_depth_path, scene_name), "*.npy")
        )
    )
    metric_depth_paths = sorted(
        glob.glob(
            os.path.join("%s/%s" % (metric_depth_path, scene_name), "*.npz")
        )
    )

    print(f"Found {len(mono_disp_paths)} mono depth files and {len(metric_depth_paths)} metric depth files for scene: {scene_name}")
    
    if len(mono_disp_paths) != len(metric_depth_paths):
        print(f"WARNING: Mismatch in number of depth files! Mono: {len(mono_disp_paths)}, Metric: {len(metric_depth_paths)}")
    img_0 = cv2.imread(image_list[0])

    scales = []
    shifts = []
    mono_disp_list = []
    fovs = []

    for t, (mono_disp_file, metric_depth_file) in enumerate(
        zip(mono_disp_paths, metric_depth_paths)
    ):
        da_disp = np.float32(np.load(mono_disp_file)) 
        if t % 20 == 0:
            print(da_disp.max(), da_disp.min())
  
        metric_depth = np.load(metric_depth_file)['depth']
        metric_K = np.load(metric_depth_file)['intrinsics']

        fov = intrinsics_to_fov(metric_K, metric_depth)
        fovs.append(fov)

        da_disp = cv2.resize(
            da_disp,
            (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        mono_disp_list.append(da_disp)
        gt_disp = 1.0 / (metric_depth + 1e-8)
        
        # avoid some bug from UniDepth
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2

        # avoid cases sky dominate entire video
        sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
        if sky_ratio > 0.5:
            non_sky_mask = da_disp > 0.01
            gt_disp_ms = (
                gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
            )
            da_disp_ms = (
                da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
            )
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
        else:
            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)

        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8

        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * da_disp)

        scales.append(scale)
        shifts.append(shift)


    print("************** UNIDEPTH FOV ", np.median(fovs))
    ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))

    K = np.eye(3)
    K[0, 0] = (
        ff * 1.0
    )  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 1] = (
        ff * 1.0
    )  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
    K[0, 2] = (
        img_0.shape[1] / 2.0
    )  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 2] = (
        img_0.shape[0] / 2.0
    )  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

    align_scale = scales[med_idx]  # np.median(np.array(scales))
    align_shift = shifts[med_idx]  # np.median(np.array(shifts))
    normalize_scale = (
        np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98)
        / 2.0
    )

    aligns = (align_scale, align_shift, normalize_scale)

    for t, mono_disp in tqdm(enumerate(mono_disp_list)):
        depth = np.clip(
            1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
            1e-4,
            1e4,
        )
        depth[depth < 1e-2] = 0.0

        basename = os.path.basename(metric_depth_paths[t].replace('.npz', ''))
        output_depth_path = os.path.join(
            save_depth_dir, basename + '.npy')
    
        np.save(output_depth_path, depth)
        output_intrinsics_path = os.path.join(
            save_K_dir, basename + '_intrinsics.npy'
        )
        np.save(output_intrinsics_path, K)



def align_davis_demo(depth_dir, data_dir, save_name='unidepth_da'):
    metric_depth_path = f'{depth_dir}/unidepthv2'
    mono_depth_path = f'{depth_dir}/depthAny_disp'

    scenes = os.listdir(mono_depth_path)

    for scene_name in scenes:
        print(scene_name)
        scene_path = f"{data_dir}/{scene_name}"

        save_depth_dir = f'{depth_dir}/{save_name}/{scene_name}'
        save_K_dir = f'{depth_dir}/{save_name}_intrinsics/{scene_name}'
        align_depth(mono_depth_path, metric_depth_path, scene_name, scene_path, save_depth_dir, save_K_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align monocular and metric depth maps')
    parser.add_argument('--depth_dir', type=str, required=True, help='Directory containing depth estimations')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--save_name', type=str, default='unidepth_da', help='Name for the saved directory')
    args = parser.parse_args()

    align_davis_demo(args.depth_dir, args.data_dir, save_name=args.save_name)