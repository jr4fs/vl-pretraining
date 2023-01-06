# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/dataset_selection/sampling/sampler.py \
    --base_path snap/vqa/LXR111/myo-sports/full_run/modelseed_965/ \
    --sampling_method min_confidence \
    --sampling_dataset myo-sports \
    --seed 965