import torch

def iproj(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=-1)
    # make intrinsics shape same as patches
    if len(intrinsics.shape) == len(patches.shape):
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    else:
        fx, fy, cx, cy = intrinsics[...,None].unbind(dim=1)

    depth = 1.0 / d.clamp(1e-2)

    xn = (x - cx) / fx * depth
    yn = (y - cy) / fy * depth

    X = torch.stack([xn, yn, depth], dim=-1)
    return X

def iproj_same_shape(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[...,None].unbind(dim=1)

    depth = 1.0 / d.clamp(1e-2)

    xn = (x - cx) / fx * depth
    yn = (y - cy) / fy * depth

    X = torch.stack([xn, yn, depth], dim=-1)
    return X

def unproj(pts_2d, intrinsics, disps):
    x, y = pts_2d.unbind(dim=2)
    d = disps
    fx = intrinsics[...,0,0]
    fy = intrinsics[...,1,1]
    cx = intrinsics[...,0,2]
    cy = intrinsics[...,1,2]

    i = torch.ones_like(disps)
    xn = (x - cx) / fx
    yn = (y - cy) / fy
    X = torch.stack([xn, yn, i, d], dim=-1)
    return X

def proj(X, intrinsics):
    """ projection """

    X, Y, Z = X.unbind(dim=-1)
    # fx = intrinsics[...,0,0][...,None]
    # fy = intrinsics[...,1,1][...,None]
    # cx = intrinsics[...,0,2][...,None]
    # cy = intrinsics[...,1,2][...,None]
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)

    d = 1.0 / Z.clamp(min=1e-2)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    return torch.stack([x,y], dim=-1)

def my_pixel2point(pixels, depth, intrinsics):
    assert pixels.size(-1) == 2, "Pixels shape incorrect"
    assert depth.size(-1) == pixels.size(-2), "Depth shape does not match pixels"
    assert intrinsics.size(-1) == intrinsics.size(-2) == 3, "Intrinsics shape incorrect."

    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    assert not torch.any(fx == 0), "fx Cannot contain zero"
    assert not torch.any(fy == 0), "fy Cannot contain zero"

    pts3d_z = depth
    pts3d_x = ((pixels[..., 0] - cx) * pts3d_z) / fx
    pts3d_y = ((pixels[..., 1] - cy) * pts3d_z) / fy
    return torch.stack([pts3d_x, pts3d_y, pts3d_z], dim=-1)