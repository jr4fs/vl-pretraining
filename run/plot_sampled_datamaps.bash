# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/dataset_selection/datamaps/plot_sampled_datamaps.py \
    --base_path snap/vqa/lxr111_multilabel_full_run_3/ \
    --sampling_ids src/dataset_selection/sampling/samples/LXR111/multilabel_full/beta/beta_kernel/tophat/seed_965/alpha_2_beta_1_budget_30.pkl
