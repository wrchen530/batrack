import os
import pickle
import argparse
import numpy as np


def vis_refined_results(result_path):
    """
    Visualize refined results from a pickle file.
    
    Args:
        result_path: Path to the results_refined.pkl file
    """
    assert os.path.exists(result_path), f"{result_path} not found"
    
    # Load results
    print(f"Loading results from {result_path}")
    results = pickle.load(open(result_path, 'rb'))
    
    # Extract data
    rgbs = results['rgbs']  # (N, H, W, 3)
    dmaps_scaled = results['dmaps_scaled']  # (N, 1, H, W) - refined depth
    cams_T_world = results['cams_T_world']  # (N, 4, 4) - camera poses
    intrinsics = results['intrinsics']  # (N, 4) - [fx, fy, cx, cy]
    
    # Print shapes
    print(f"\nData shapes:")
    print(f"  rgbs: {np.array(rgbs).shape}")
    print(f"  dmaps_scaled (refined depth): {np.array(dmaps_scaled).shape}")
    print(f"  cams_T_world (poses): {np.array(cams_T_world).shape}")
    print(f"  intrinsics: {np.array(intrinsics).shape}")
    
    # Convert to appropriate formats for visualization
    image_list = []
    depth_list_scaled = []
    intrinsics_list = []
    
    for i in range(len(rgbs)):
        image_list.append(rgbs[i])  # Already in (H, W, 3) format
        depth_list_scaled.append(dmaps_scaled[i].squeeze()[..., None])  # (H, W, 1)
        
        # Convert intrinsics from [fx, fy, cx, cy] to 3x3 matrix
        fx, fy, cx, cy = intrinsics[i]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics_list.append(K)
    
    print(f"\nVisualization data prepared:")
    print(f"  Number of frames: {len(image_list)}")
    print(f"  Image shape: {image_list[0].shape}")
    print(f"  Depth shape: {depth_list_scaled[0].shape}")
    print(f"  Intrinsics shape: {intrinsics_list[0].shape}")
    
    # Visualize using rerun
    print(f"\nStarting visualization...")
    from main.rerun_visualizer import vis_rerun_demo

    vis_rerun_demo(
        results, 
        results_total=None, 
        image_list=image_list, 
        intrinsics_list=intrinsics_list, 
        depth_list_scaled=depth_list_scaled,  # Refined depth
        depth_filter_th=0.05, 
        image_plane_distance=0.1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize refined results from pickle file')
    parser.add_argument('--result_path', type=str, 
                        default='results_refined.pkl',
                        help='Path to the results_refined.pkl file')
    
    args = parser.parse_args()
    vis_refined_results(args.result_path)
