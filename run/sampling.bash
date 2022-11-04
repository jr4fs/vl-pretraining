# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/dataset_selection/sampling/sampler.py \
    --base_path snap/vqa/vqa_lxr111_animals_fromScratch_20epochs_breeds/ \
    --sampling_method beta \
    