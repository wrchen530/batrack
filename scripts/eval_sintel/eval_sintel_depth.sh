RESULT_DIR=outputs/sintel/zoedepth_nk

python main/global_refine/eval_sintel_depth.py \
--result_dir $RESULT_DIR \
--grid_size 8 \
--niter 300 \
