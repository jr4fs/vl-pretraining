# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert
name=$2
IDS=


# Save logs and models under snap/vqa; make backup.
# output=snap/vqa/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_optuna.py \
    --train train,nominival --valid minival  \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --fromScratch \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 20 \
    --tqdm --subset animals #--output $output ${@:3}

# while [ "$1" != "" ]; do
#     case $1 in
#     --sampling_ids)
#         IDS=$2
#         ;;
#     esac
#     shift
# done

# if [[ "$IDS" != "" ]]; then
#     FILENAME=${IDS##*/}
#     cp $IDS $output/$FILENAME
# fi