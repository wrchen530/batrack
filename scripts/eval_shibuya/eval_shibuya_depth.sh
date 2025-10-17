RESULT_DIR=outputs/shibuya/zoedepth_nk

python main/global_refine/eval_shibuya_depth.py \
--result_dir $RESULT_DIR \
--grid_size 8 \
--niter 300 \