DATASET=data/shibuya
DEPTHDIR=data/Monodepth/shibuya
SAVEDIR=outputs/shibuya/zoedepth_nk


mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in Standing01 Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07
do
    python main/run_batrack.py \
    --config-path=../configs \
    --config-name=shibuya \
    data.imagedir=$DATASET/$SCENE/image_0 \
    data.gt_traj=$DATASET/$SCENE/gt_pose.txt \
    data.savedir=$SAVEDIR \
    +data.depthdir=$DEPTHDIR/zoedepth_nk/$SCENE \
    +data.depthdir_gt=$DATASET/$SCENE/depth_0 \
    data.calib=calibs/tartan_shibuya.txt \
    data.name=$SCENE \
    save_trajectory=true \
    save_plot=true \
    save_results=true
    
done


