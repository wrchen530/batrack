DATASET=data/sintel/training
DEPTHDIR=data/Monodepth/sintel
SAVEDIR=outputs/sintel/zoedepth_nk


mkdir -p $SAVEDIR

CONFIG=sintel

cp configs/${CONFIG}.yaml $SAVEDIR/config.yaml
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
do
    SCENE_PATH=$DATASET/final/$SCENE
    python main/run_batrack.py \
    --config-path=../configs \
    --config-name=$CONFIG \
    data.imagedir=$SCENE_PATH \
    +data.depthdir=$DEPTHDIR/zoedepth_nk/$SCENE \
    +data.depthdir_gt=$DATASET/depth/$SCENE \
    data.gt_traj=$DATASET/camdata_left/$SCENE \
    data.savedir=$SAVEDIR \
    data.calib=$DATASET/camdata_left/$SCENE \
    data.name=$SCENE \
    save_trajectory=true \
    save_plot=true \
    save_results=true \
    
done



