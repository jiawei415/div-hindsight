#!/bin/sh

cuda_id=$1
seed=$2
env_name=$3

rollout_num=2
k_heads=1
num_cpu=19


logdir="~/results/her"
# logdir="/root/gpu_ceph/ztjiaweixu/her"

time=2
for i in $(seq 1)
do
    export CUDA_VISIBLE_DEVICES=$cuda_id
    tag=$(date "+%Y%m%d%H%M%S")
    if [ $(echo $env_name | grep "Fetch")x != ""x ];then
        echo "Fetch"
        python -m baselines.her.experiment.train --env_name ${env_name} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --num_cpu ${num_cpu} --clip_div 0.001 \
        --logdir $logdir > ~/logs/${env_name}_${tag}.out 2> ~/logs/${env_name}_${tag}.err &
    elif [ $(echo $env_name | grep "Rotate")x != ""x ];then
        echo "Rotate"
        python -m baselines.her.experiment.train --env_name ${env_name} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --num_cpu ${num_cpu} --goal_type rotate --sigma 0.1 \
        --logdir $logdir > ~/logs/${env_name}_${tag}.out 2> ~/logs/${env_name}_${tag}.err &
    elif [ $(echo $env_name | grep "Full")x != ""x ];then
        echo "Full"
        python -m baselines.her.experiment.train --env_name ${env_name} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} --num_cpu ${num_cpu} --goal_type full --sigma 0.1 \
        --logdir $logdir > ~/logs/${env_name}_${tag}.out 2> ~/logs/${env_name}_${tag}.err &
    else
        echo "$env_name"
        python -m baselines.her.experiment.train --env_name ${env_name} --seed ${seed} --k_heads ${k_heads} --rollout_num ${rollout_num} \
        --logdir $logdir > ~/logs/${env_name}_${tag}.out 2> ~/logs/${env_name}_${tag}.err &
    fi
    echo "run $env_name $seed $tag"
    let seed=$seed+1
    let cuda_id=$cuda_id+1
    sleep ${time}
done

# ps -ef | grep baselines | awk '{print $2}'| xargs kill -9

# MazeA-v1, MazeB-v1, MazeC-v1, MazeD-v1
# PointMassEmptyEnv-v1, PointMassWallEnv-v1, PointMassRoomsEnv-v1, Point2DLargeEnv-v1, Point2DFourRoom-v1
# SawyerReachXYEnv-v1, SawyerDoorFixEnv-v1, SawyerDoorAngle-v1, SawyerDoorPos-v1
# FetchSlide-v1, FetchPush-v1, FetchPickAndPlace-v1, FetchReach-v1, Reacher-v2, HandReach-v0
# HandManipulateBlock-v0, HandManipulateBlockFull-v0, HandManipulateBlockRotateParallel-v0, HandManipulateBlockRotateXYZ-v0, HandManipulateBlockRotateZ-v0
# HandManipulateEgg-v0, HandManipulateEggFull-v0, HandManipulateEggRotate-v0
# HandManipulatePen-v0, HandManipulatePenFull-v0, HandManipulatePenRotate-v0
