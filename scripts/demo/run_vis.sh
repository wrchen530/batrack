#!/bin/bash

SCENE=tennis
RESULT_PATH=outputs/davis/unidepth_dav2/$SCENE/results_refined.pkl

python main/vis_refined_results.py \
    --result_path $RESULT_PATH
