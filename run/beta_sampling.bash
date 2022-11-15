seed=$1
gpu=$2

bash run/vqa_from_scratch.bash $gpu LXR111/animals/beta/beta_pvals/alpha_2_beta_2_budget_10_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_pvals/seed_$seed/alpha_2_beta_2_budget_10.pkl --sampling_method beta --neptune_study_name beta_22_pvals_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/beta/beta_pvals/alpha_2_beta_2_budget_20_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_pvals/seed_$seed/alpha_2_beta_2_budget_20.pkl --sampling_method beta --neptune_study_name beta_22_pvals_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/beta/beta_pvals/alpha_2_beta_2_budget_30_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_pvals/seed_$seed/alpha_2_beta_2_budget_30.pkl --sampling_method beta --neptune_study_name beta_22_pvals_seed$seed