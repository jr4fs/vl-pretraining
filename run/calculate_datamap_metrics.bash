# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/dataset_selection/datamaps/calculate_datamap_stats.py \
    --base_path snap/vqa/lxr111_multilabel_full_run_3/
