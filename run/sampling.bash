# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/dataset_selection/sampling/sampler.py \
    --base_path snap/vqa/lxr111_multilabel_full_run_3/ \
    --sampling_method beta_multilabel \
    --sampling_dataset multilabel_full \
    --seed 965