DATASET=data/davis
DEPTHDIR=data/Monodepth/davis
SAVEDIR=outputs/davis/unidepth_dav2

CONFIG=davis_demo

for SCENE in tennis
do
    python main/run_batrack.py \
    --config-path=../configs \
    --config-name=$CONFIG \
    data.imagedir=$DATASET/$SCENE \
    data.savedir=$SAVEDIR \
    +data.depthdir=$DEPTHDIR/unidepth_dav2/$SCENE \
    +data.depthdir_gt=$DEPTHDIR/unidepth_dav2/$SCENE \
    data.calib=$DEPTHDIR/unidepth_dav2_intrinsics/$SCENE \
    data.name=$SCENE \
    save_trajectory=true \
    save_plot=true \
    save_results=true \

done
