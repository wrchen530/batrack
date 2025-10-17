#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


evalset=(
  tennis
)

DATA_DIR=data/davis
DEPTH_DIR=data/Monodepth/davis

mkdir -p $DEPTH_DIR
# Run DepthAnything
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos_v2.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_v2_vitl.pth \
  --img-path $DATA_DIR/$seq \
  --outdir $DEPTH_DIR/depthAny_disp/$seq/
done

# Run DepthAnythingV2
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos_v2.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_v2_vitl.pth \
  --img-path $DATA_DIR/$seq \
  --outdir $DEPTH_DIR/depthAny_disp/$seq/
done


# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_DIR/$seq \
  --outdir $DEPTH_DIR/unidepthv2
done


# align DA with UniDepth
python main/mono_depth/get_mono_depth.py \
--data_dir $DATA_DIR \
--depth_dir $DEPTH_DIR \
--save_name unidepth_dav2