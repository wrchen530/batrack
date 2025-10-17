# Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction

**[ICCV2025, Oral]** This repository contains the official implementation of [BA-Track](https://wrchen530.github.io/projects/batrack/). Our method achieves dynamic scene reconstruction via motion decoupling, bundle adjustment, and global refinement.

> **Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction**<br>
> [Weirong Chen](https://wrchen530.github.io/), [Ganlin Zhang](https://ganlinzhang.xyz/), [Felix Wimbauer](https://fwmb.github.io/), [Rui Wang](https://rui2016.github.io/), [Nikita Araslanov](https://arnike.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Daniel Cremers](https://cvg.cit.tum.de/members/cremers) <br>
> ICCV 2025

**[[Paper](https://arxiv.org/abs/2504.14516)] [[Project Page](https://wrchen530.github.io/projects/batrack/)]**

<div align="center">
  <p align="center">
  <a href="">
    <img src="./assets/demo.gif" alt="Demo" width="70%">
  </a>
</p>
</div>

## Todo
- [x] Initial release with demo
- [x] Release pre-trained checkpoints
- [x] Add scripts for evaluation
- [ ] Add visualization for motion decoupling
- [ ] Add scripts for training data preparation

## Setting Up the Environment

### Requirements
The code was tested on Ubuntu 22.04, PyTorch 2.1.1, and CUDA 11.8 with an NVIDIA A40. Follow the steps below to set up the environment.

### Clone the repository
```
git clone https://github.com/wrchen530/batrack.git
cd batrack
```

### Create a conda environment and install dependencies
```
conda env create -f environment.yml
conda activate batrack
pip install -r requirements.txt
```

### Install the batrack package
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install .
```

### Install xformers for UniDepth
To install xformers for the UniDepth model, follow the instructions at https://github.com/facebookresearch/xformers. If you encounter installation issues, we recommend installing from a prebuilt package. For example, for Python 3.10 + CUDA 11.8 + PyTorch 2.1.1:
```
wget https://anaconda.org/xformers/xformers/0.0.23/download/linux-64/xformers-0.0.23-py310_cu11.8.0_pyt2.1.1.tar.bz2
conda install xformers-0.0.23-py310_cu11.8.0_pyt2.1.1.tar.bz2
```

## Demo with DAVIS
We follow [MegaSAM](https://github.com/mega-sam/mega-sam) to extract monocular depth priors from UniDepthV2 and DepthAnythingV2. Then we run our method in two stages: (1) sparse SLAM and (2) dense global alignment.

### Download sample sequence
- Download sample DAVIS sequence from [Google Drive](https://drive.google.com/file/d/1hlHyxqW0AaPrv6NwJya0P2UJpcVNYbBU/view?usp=drive_link) and save it to `data/davis`.

### Download checkpoints
- Download the DepthAnythingV2 checkpoint from [this link](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth) and save it to `batrack/Depth-Anything/checkpoints/depth_anything_v2_vitl.pth`.
- Download our tracker checkpoint from [Google Drive](https://drive.google.com/file/d/1wWK_ur0Pr4jivqDUdyRUFzHPaF-f1clC/view?usp=sharing) and save it to `batrack/checkpoints/md_tracker.pth`.

### Step 1: Monocular Depth Estimation
Compute monocular depth priors from UniDepthV2 and DepthAnythingV2, and align their scales:
```
bash scripts/demo/run_mono_depth.sh
```

### Step 2: Sparse SLAM
Run the sparse SLAM pipeline to perform motion decoupling and bundle adjustment for pose estimation and initial sparse reconstruction:
```
bash scripts/demo/run_sparse.sh
```

### Step 3: Dense Global Alignment
Perform dense global alignment to refine the reconstruction using monocular depth priors:
```
bash scripts/demo/run_dense.sh
```

### Step 4: Visualization (Optional)
Visualize reconstruction results with Rerun:
```
bash scripts/demo/run_vis.sh
```

## Evaluations
We provide evaluation scripts for MPI-Sintel and TartanAir-Shibuya.

### MPI-Sintel
Download MPI-Sintel from [MPI-Sintel](http://sintel.is.tue.mpg.de/) and place it in the `data` folder at `data/sintel`. For evaluation, also download the [ground-truth camera pose data](http://sintel.is.tue.mpg.de/depth). The folder structure should look like:
```
sintel
└── training
    ├── final
    └── camdata_left
```

**Precomputed depths.** To avoid environment/dependency conflicts, we provide precomputed ZoeDepth results at [this link](https://drive.google.com/file/d/1y8zPOMlwRzeP43RBKgA6gg8-_EjlurCy/view?usp=drive_link). Download and place the folder at `data/Monodepth/sintel/zoedepth_nk`.

Run pose evaluation:
```
bash scripts/eval_sintel/eval_sintel_pose.sh
```
Run depth evaluation:
```
bash scripts/eval_sintel/eval_sintel_depth.sh
```

### TartanAir-Shibuya
Download TartanAir-Shibuya following the instructions at [TartanAir-Shibuya](https://github.com/haleqiu/tartanair-shibuya) and place it in the `data` folder at `data/shibuya`.

For `RoadCrossing07/image_0`, skip the first 5 images (000000.png to 000004.png) because there is no depth ground truth. You can delete these files with:
```bash
# Delete first 5 images (000000.png to 000004.png) for RoadCrossing07/image_0
rm data/shibuya/RoadCrossing07/image_0/00000{0,1,2,3,4}.png
```

**Precomputed depths.** To avoid environment/dependency conflicts, we provide precomputed ZoeDepth results at [this link](https://drive.google.com/file/d/14XHNH9WNDf3fMm5rGNDH-NP1n00eUpHF/view?usp=drive_link). Download and place the folder at `data/Monodepth/shibuya/zoedepth_nk`.

Run pose evaluation:
```
bash scripts/eval_shibuya/eval_shibuya_pose.sh
```
Run depth evaluation:
```
bash scripts/eval_shibuya/eval_shibuya_depth.sh
```

## Citations
If you find this repository useful, please consider citing our paper:
```
@InProceedings{chen2025back,
  title={Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction},
  author={Chen, Weirong and Zhang, Ganlin and Wimbauer, Felix and Wang, Rui and Araslanov, Nikita and Vedaldi, Andrea and Cremers, Daniel},
  journal={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Acknowledgements
We adapted code from several excellent repositories, including:
- [CoTracker](https://github.com/facebookresearch/co-tracker)
- [SpaTracker](https://github.com/henry123-boy/SpaTracker)
- [DPVO](https://github.com/princeton-vl/DPVO)
- [LEAP-VO](https://github.com/wrchen530/leapvo)
- [MegaSAM](https://github.com/mega-sam/mega-sam)

We sincerely thank the authors for open-sourcing their work.

## Concurrent Efforts
Several exciting concurrent works explore related aspects of dynamic scene reconstruction and point tracking! Check them out:
- **[SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2)** - SpatialTrackerV2: 3D Point Tracking Made Easy
- **[MVTracker](https://github.com/ethz-vlg/mvtracker)** - Multi-View 3D Point Tracking
- **[C4D](https://littlepure2333.github.io/C4D/)** - C4D: 4D Made from 3D through Dual Correspondences

## Limitations
This project attempts to disentangle camera-induced and object motion via point tracking. The model was trained on a relatively small, domain-specific dataset (Kubric), which may limit its generalization to challenging or novel scenes. Future directions include expanding the training data and refining the tracker architecture to improve robustness and efficiency.