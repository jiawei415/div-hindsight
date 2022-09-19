#!/bin/sh
envname=$1
seed=$2
rollout_num=1
k_heads=16

export CUDA_VISIBLE_DEVICES=$3

logdir="~/results/her"
# logdir="/root/gpu_ceph/ztjiaweixu/her"

time=2
for i in $(seq 1)
do
    tag=$(date "+%Y%m%d%H%M%S")
    if [ $(echo $envname | grep "Fetch")x != ""x ];then
        echo "Fetch"
        python -m baselines.her.experiment.train --env_name ${envname} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --clip_div 0.001 \
        --logdir $logdir > ~/logs/${envname}_${tag}.out 2> ~/logs/${envname}_${tag}.err &
    elif [ $(echo $envname | grep "Rotate")x != ""x ];then
        echo "Rotate"
        python -m baselines.her.experiment.train --env_name ${envname} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --goal_type rotate --sigma 0.1 \
        --logdir $logdir > ~/logs/${envname}_${tag}.out 2> ~/logs/${envname}_${tag}.err &
    elif [ $(echo $envname | grep "Full")x != ""x ];then
        echo "Full"
        python -m baselines.her.experiment.train --env_name ${envname} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --goal_type full --sigma 0.1 \
        --logdir $logdir > ~/logs/${envname}_${tag}.out 2> ~/logs/${envname}_${tag}.err &
    else
        echo "$env_name"
        python -m baselines.her.experiment.train --env_name ${envname} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} \
        # --logdir $logdir > ~/logs/${envname}_${tag}.out 2> ~/logs/${envname}_${tag}.err &
    fi
    echo "run $envname $seed $tag"
    let seed=$seed+1
    sleep ${time}
done

# ps -ef | grep ${envname} | awk '{print $2}'| xargs kill -9
# ps -ef | grep baselines | awk '{print $2}'| xargs kill -9

# MazeA-v1, MazeB-v1, MazeC-v1, MazeD-v1
# PointMassEmptyEnv-v1, PointMassWallEnv-v1, PointMassRoomsEnv-v1, Point2DLargeEnv-v1, Point2DFourRoom-v1
# SawyerReachXYEnv-v1, SawyerDoorFixEnv-v1, SawyerDoorAngle-v1, SawyerDoorPos-v1
# FetchSlide-v1, FetchPush-v1, FetchPickAndPlace-v1, FetchReach-v1, Reacher-v2, HandReach-v0
# HandManipulateBlock-v0, HandManipulateBlockFull-v0, HandManipulateBlockRotateParallel-v0, HandManipulateBlockRotateXYZ-v0, HandManipulateBlockRotateZ-v0
# HandManipulateEgg-v0, HandManipulateEggFull-v0, HandManipulateEggRotate-v0
# HandManipulatePen-v0, HandManipulatePenFull-v0, HandManipulatePenRotate-v0
