
RESULT_DIR=outputs/davis/unidepth_dav2

for scene in tennis 
do
    echo "Processing scene: $scene"
    python main/global_refine/run_global_refine.py \
        --result_dir $RESULT_DIR \
        --grid_size 12 \
        --scenes $scene \
        --niter 300 \
        
done