# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert
name=$2
IDS=

# bash run/vqa_from_scratch.bash <GPU_ID> <name of folder to save run> --subset <animals, sports> --sampling_ids <src/dataset_selection/sampling/samples> 
# bash run/vqa_from_scratch.bash 0 LXR111/animals/random/test/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/random/budget_10.pkl 
# Save logs and models under snap/vqa; make backup.
output=snap/vqa_pretrained/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --loadLXMERTQA snap/pretrained/model \
    --fromScratch \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 22 \
    --tqdm --output $output ${@:3}

while [ "$1" != "" ]; do
    case $1 in
    --sampling_ids)
        IDS=$2
        ;;
    esac
    shift
done

if [[ "$IDS" != "" ]]; then
    FILENAME=${IDS##*/}
    cp $IDS $output/$FILENAME
fi