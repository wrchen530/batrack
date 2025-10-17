import rerun as rr
import numpy as np
import torch
from einops import repeat, rearrange

def reproject(cams_params, trajs_2d, points_depth, cam_T_world=None):
    """
    cams_params (3,3)
    cam_T_world (4,4)
    trajs_2d (N, 2)
    points_depth (N, )
    """
    N = trajs_2d.shape[0]
    trajs_3d = np.concatenate([trajs_2d, np.ones_like(points_depth)[...,None]], axis=1)
    trajs_3d = trajs_3d.reshape(N, 3, 1)
    K_inv = np.linalg.inv(cams_params)
    K_inv = repeat(K_inv, 'c1 c2 -> n c1 c2', n=N)
    trajs_3d = K_inv @ trajs_3d
    trajs_3d = trajs_3d * points_depth.reshape(N, 1, 1)
    
    if cam_T_world is not None:
        cam_T_world = repeat(cam_T_world, 'c1 c2 -> n c1 c2', n=N)
        R = cam_T_world[:, :3 ,:3]
        tvec = cam_T_world[:, :3, [3]]
        trajs_3d_world = R @ trajs_3d + tvec
        trajs_3d_world = trajs_3d_world.reshape(N, 3)
        return trajs_3d_world 
    else:
        return trajs_3d[...,0]

def get_grid_xy(h, w, homogeneous=False):
    """
    Generate a grid of (x, y) coordinates for an image of size (h, w).
    If homogeneous is True, an extra row of ones is added.
    """
    # Create meshgrid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Flatten the coordinates
    x = x.flatten()
    y = y.flatten()
    
    # Stack x and y into a (2, N) tensor where N = h * w
    grid = torch.stack((x, y), dim=0).float()
    
    if homogeneous:
        # Add a row of ones for homogeneous coordinates
        ones = torch.ones(1, h * w)
        grid = torch.cat((grid, ones), dim=0)
    
    return grid



def lift_image(img, depth, pose, proj):

    depth_torch = torch.tensor(depth, device='cpu').float()
    pose_torch = torch.tensor(pose, device='cpu').float()

    h, w = img.shape[:2]
    device = 'cpu'
    proj = torch.tensor(proj, device=device).float()

    proj[0, 0] = proj[0, 0] / w * 2
    proj[1, 1] = proj[1, 1] / h * 2
    proj[0, 2] = proj[0, 2] / w * 2 - 1
    proj[1, 2] = proj[1, 2] / h * 2 - 1

    inv_proj = torch.inverse(proj)

    pts = get_grid_xy(h, w, homogeneous=True).reshape(3, h*w)
    pts = inv_proj @ pts
    pts = pts * depth_torch.view(1, -1)
    pts = torch.cat((pts, torch.ones(1, h*w, device=device)), dim=0)
    pts = pose_torch @ pts
    pts = pts[:3, :].T
    colors = torch.tensor(img.reshape(-1, 3))
    pts = pts.cpu().numpy()
    colors = colors.cpu().numpy()
    return pts, colors



def generate_point_cloud(img, depth, intrinsics, pose):
    """
    Generate a dense point cloud from an image, depth map, intrinsics, and camera pose.
    
    Parameters:
    - img: (H, W, 3) RGB image
    - depth: (H, W) depth map
    - intrinsics: (3, 3) camera intrinsic matrix
    - pose: (4, 4) camera-to-world transformation matrix

    Returns:
    - pts_world: (N, 3) array of 3D points in world coordinates
    - colors: (N, 3) array of RGB colors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to the device
    img = torch.tensor(img, device=device).float()
    depth = torch.tensor(depth, device=device).float()
    intrinsics = torch.tensor(intrinsics, device=device).float()
    pose = torch.tensor(pose, device=device).float()
    
    H, W = depth.shape

    # Step 1: Create a grid of (x, y) coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Step 2: Convert pixel coordinates to normalized camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert to normalized coordinates
    z = depth.reshape(-1)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Step 3: Convert to homogeneous coordinates in the camera frame
    pts_camera = torch.stack((x, y, z, torch.ones_like(z)), dim=0)  # Shape: (4, H*W)

    # Step 4: Transform points to world coordinates using the pose matrix
    pts_world = (pose @ pts_camera)[:3, :].T  # Shape: (H*W, 3)

    # Step 5: Get corresponding colors
    colors = img.reshape(-1, 3)  # Shape: (H*W, 3)

    return pts_world.cpu().numpy(), colors.cpu().numpy()


def filter_depth(depth_np, threshold=0.1):
    depth_np = rearrange(depth_np, 'h w c -> c h w')
    _, h, w = depth_np.shape
    depth = torch.from_numpy(depth_np)


    depth = depth.clone()[None, ...]
    
    depth_grad = torch.stack(torch.gradient(depth, dim=(-2, -1))).norm(dim=0)

    depth[depth_grad > threshold] = 0

    depth_np_new = rearrange(depth[0].numpy(), 'c h w -> h w c')
    return depth_np_new




def vis_rerun_demo(results, results_total, image_list, intrinsics_list, depth_list_scaled=None, depth_filter_th=0.0, image_plane_distance=1.0):
    cams_T_world = results['cams_T_world']
    tstamps = results['tstamps']
    grid_query_frames = results['grid_query_frames']
    trajs_valid = results['trajs_valid']
    trajs_static = results['trajs_static']
    trajs_vis = results['trajs_vis']


    trajs_2d_disp = results['trajs_2d_disp']

    if results_total is not None:
        trajs_2d_disp_total = results_total['trajs_2d_disp']
    else:
        trajs_2d_disp_total = None

    S = tstamps.shape[0]

    H, W, _ = image_list[0].shape

    # rr.init("batrack", spawn=False)
    # rr.connect("131.159.19.97:9876")
    rr.init("batrack", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)


    for s in range(S-1):

        rr.set_time_sequence("frame", s)

        K = intrinsics_list[s]
        img_rgb = image_list[s]
        cam_T_world = cams_T_world[s]


        rr.log("world/scene/cam_traj", rr.LineStrips3D([pose[:3, 3] for pose in cams_T_world[:s]], colors=[(0, 255, 0)]), static=False)

        if True:
            rr.log(
            f"world/camera",
                rr.Pinhole(
                    resolution=[W, H],
                    image_from_camera=K,
                    # camera_xyz=rr.ViewCoordinates.RDF,  # FIXME LUF -> RDF
                    image_plane_distance=image_plane_distance,
                ),
            )
            rr.log(
                f"world/camera",
                rr.Transform3D(translation=cam_T_world[:3,3], mat3x3=cam_T_world[:3,:3]),

            )

            rr.log(f"world/camera/rgb", rr.Image(img_rgb, opacity=1.0))


            if depth_list_scaled is not None:
                
                img_depth_scaled = depth_list_scaled[s]

                img_depth_scaled_filtered = filter_depth(img_depth_scaled, threshold=depth_filter_th) 
                


                H, W, _ = img_rgb.shape
                margin = int(0.05 * H)
                img_rgb_crop = img_rgb[margin:-margin, margin:-margin]
                img_depth_scaled_filtered_crop = img_depth_scaled_filtered[margin:-margin, margin:-margin]
                # modify K
                K_ = K.copy()
                K_[0,2] -= margin   
                K_[1,2] -= margin
                pts, colors = generate_point_cloud(img_rgb_crop, img_depth_scaled_filtered_crop[...,0], K_, cam_T_world)

                colors = colors.astype(np.uint8)

                rr.log(f"world/pointcloud_refined", rr.Points3D(pts, colors=colors))


    
    N = grid_query_frames.shape[0]
    S_local = trajs_2d_disp.shape[2]

    window_size = (S_local + 1) // 2
    print("N", N, "S_local", S_local, "window_size", window_size)
    for i in range(N):
        query_t = grid_query_frames[i].astype(int)

        for s in range(S_local):
            frame_t = query_t - window_size + s + 1

            if frame_t < 0 or frame_t >= S:
                continue
           
            rr.set_time_sequence("frame", frame_t)

            traj_2d = trajs_2d_disp[query_t, :, s, :2]
            

            vis_mask = trajs_vis[query_t, :, s] > 0.5
            is_static = trajs_static[query_t, :, s] > 0.5


            static_2d = traj_2d[is_static]
            dyn_2d = traj_2d[(~is_static)]

            rr.log(f"world/camera/image/point2d_static/query_{i}",  rr.Points2D(static_2d,  colors=[0,255,0], radii=2))
            rr.log(f"world/camera/image/point2d_dyn/query_{i}",  rr.Points2D(dyn_2d,  colors=[255,0,0], radii=2)) 

            if trajs_2d_disp_total is not None:
                traj_2d_total = trajs_2d_disp_total[query_t, :, s, :2]
                dyn_2d_total = traj_2d_total[(~is_static)]
                rr.log(f"world/camera/image/point2d_dyn_total/query_{i}",  rr.Points2D(dyn_2d_total,  colors=[255,255,0], radii=2))
   
            K = intrinsics_list[frame_t]

        rr.set_time_sequence("frame", query_t + window_size)
        rr.log(f"world/camera/image/point2d_static/query_{i}", rr.Clear(recursive=False))
        rr.log(f"world/camera/image/point2d_dyn/query_{i}", rr.Clear(recursive=False))
        rr.log(f"world/camera/image/point2d_dyn_total/query_{i}", rr.Clear(recursive=False))
   


