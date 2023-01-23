# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --multilabel \
    --tqdm --output $output ${@:3}

# # The name of this experiment.
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate lxmert
# name=$1

# # Save logs and models under snap/vqa; make backup.
# output=snap/vqa/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash

# # See Readme.md for option details.
# PYTHONPATH=$PYTHONPATH:./src \
#     python src/tasks/vqa.py \
#     --train train,nominival --valid minival  \
#     --llayers 9 --xlayers 5 --rlayers 5 \
#     --loadLXMERTQA snap/pretrained/model \
#     --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
#     --tqdm --output $output ${@:3}
