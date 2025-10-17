import os
import cv2
import numpy as np
from pathlib import Path
from itertools import chain
from PIL import Image

RUN_REPLICA = True
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_depth(filename):
    depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED) 
    depth = depth / 1000. # turn depth from mm to meter
    return depth


def cam_read_sintel(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def sintel_stream(imagedir, calib_root, stride, skip=0):
    """ image generator """

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camfile = str(imfile).split('/')[-1].replace('.png','.cam')
        K, _ = cam_read_sintel(os.path.join(calib_root, camfile))
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        calib = [fx, fy, cx, cy]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        yield (t, image, intrinsics)

    yield (-1, image, intrinsics) 

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
scalingFactor = 5000.0

def tum_depth_read(depth_file):
    depth = Image.open(depth_file)
    Z = np.array(depth) / scalingFactor
    return Z


def sintel_depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def load_depth_file(filename, mode='sintel'):
    if '.npy' in filename:
        depth = np.load(filename)
    elif '.npz' in filename:
        depth = np.load(filename)['depth']
    else:
        if mode == 'sintel':
            depth = sintel_depth_read(filename)
        elif mode == 'tum':
            depth = tum_depth_read(filename)
        
    if len(depth.shape) == 2:
        depth = depth[...,None]
    return depth.astype(float)


def sintel_rgbd_stream(imagedir, depthdir, depthdir_gt, calib_root, stride, skip=0, end=-1, input_intrinsics=False):
    """ image generator """

    print("input_intrinsics", input_intrinsics)

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    depth_exts = ["*.npy", "*.npz"]
    depth_exts_gt = ["*.dpt"]

    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))[skip::stride]
    depth_list_gt = sorted(chain.from_iterable(Path(depthdir_gt).glob(e) for e in depth_exts_gt))[skip::stride]

    if input_intrinsics:
        K_exts = ["*.npy"]
        K_list = sorted(chain.from_iterable(Path(calib_root).glob(e) for e in K_exts))
        Ks = np.array([np.load(str(k)) for k in K_list])

        if end == -1:
            end = len(image_list)

        Ks = Ks[skip:end:stride]


    print("imagedir", imagedir)
    print("depthdir", depthdir)
    print("depthdir_gt", depthdir_gt)
    assert len(depth_list) == len(image_list), f"depth_list: {len(depth_list)}, image_list: {len(image_list)}"

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if input_intrinsics:
            K = Ks[0] # only one camera intrinsics
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            calib = [fx, fy, cx, cy]
        else:
            camfile = str(imfile).split('/')[-1].replace('.png','.cam')
            K, _ = cam_read_sintel(os.path.join(calib_root, camfile))
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            calib = [fx, fy, cx, cy]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
    
        depth = load_depth_file(str(depth_list[t]))
        depth = depth[:h-h%16, :w-w%16]

        # if shape mismatch, use nearest resize
        if depth.shape[0] != image.shape[0] or depth.shape[1] != image.shape[1]:
            print("depth shape mismatch", depth.shape, image.shape)
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = depth[..., None]
            
        depth_gt = load_depth_file(str(depth_list_gt[t]), mode='sintel')
        depth_gt = depth_gt[:h-h%16, :w-w%16]

        yield (t, image, depth, depth_gt, intrinsics)

    yield (-1, image, depth, depth_gt, intrinsics) 


def tartanair_rgbd_stream(imagedir, depthdir, depthdir_gt, calib_root, stride, skip=0, end=-1):
    """ image generator """
    calib = np.loadtxt(calib_root, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
    
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    depth_exts = ["*.npy", "*.npz"]
    depth_exts_gt = ["*.npy"]

    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))[skip::stride]
    depth_list_gt = sorted(chain.from_iterable(Path(depthdir_gt).glob(e) for e in depth_exts_gt))[skip::stride]

    assert len(depth_list) == len(image_list), f"depth_list: {len(depth_list)}, image_list: {len(image_list)}"
    assert len(depth_list_gt) == len(image_list), f"depth_list_gt: {len(depth_list_gt)}, image_list: {len(image_list)}"
    for t, imfile in enumerate(image_list):

        timestamp = str(imfile).split('/')[-1].replace('.png', '')

        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
    
        depth = load_depth_file(str(depth_list[t]))
        depth = depth[:h-h%16, :w-w%16]

        depth_gt = load_depth_file(str(depth_list_gt[t]), mode='tartanair')
        depth_gt = depth_gt[:h-h%16, :w-w%16]
        # FIXME: KITTI depth is not used

        yield (timestamp, image, depth, depth_gt, intrinsics)

    yield (-1, image, depth, depth_gt, intrinsics) 




def davis_stream(imagedir, depthdir, calib_root, stride, skip=0, end=-1):

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))

    depth_exts = ["*.npy"]

    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))



    K_exts = ["*.npy"]
    K_list = sorted(chain.from_iterable(Path(calib_root).glob(e) for e in K_exts))
    Ks = np.array([np.load(str(k)) for k in K_list])

    if end == -1:
        end = len(image_list)

    Ks = Ks[skip:end:stride]
    
    
    image_list = image_list[skip:end:stride]
    depth_list = depth_list[skip:end:stride]

    assert len(depth_list) == len(image_list), f"depth_list: {len(depth_list)}, image_list: {len(image_list)}"
    assert Ks.shape[0] == len(image_list), f"Ks: {Ks.shape[0]}, image_list: {len(image_list)}"

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        fx, fy, cx, cy = Ks[t,0,0], Ks[t,1,1], Ks[t,0,2], Ks[t,1,2]

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
    
        depth = load_depth_file(str(depth_list[t]))
        depth = depth[:h-h%16, :w-w%16]

        yield (t, image, depth, depth, intrinsics)
        # yield (t, image, depth_gt, depth_gt, intrinsics)

    yield (-1, image, depth, depth, intrinsics) 
    # yield (-1, image, depth_gt, depth_gt, intrinsics) 
   


def dataset_rgbd_stream(imagedir, depthdir, calib, stride, skip=0, mode='replica'):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    
    depth_exts = ["*.npy"]

    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))[skip::stride]
    print(depthdir)
    assert len(depth_list) == len(image_list), f"depth_list: {len(depth_list)}, image_list: {len(image_list)}"

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        depth = load_depth_file(str(depth_list[t]))
        depth = depth[:h-h%16, :w-w%16]


        yield (t, image, depth, depth, intrinsics)

    yield (-1, image, depth, depth, intrinsics)



