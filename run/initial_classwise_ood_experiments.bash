bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/full_run/ --subset animals --neptune_study_name full_run_ood --training_budget 100

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/random/budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/random/seed_965/budget_30.pkl --sampling_method random --neptune_study_name classwise_random_seed965_ood --training_budget 30

# bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/max_confidence/budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/max_confidence/seed_965/budget_30.pkl --sampling_method max_confidence --neptune_study_name classwise_max_confidence_seed965_ood --training_budget 30

# bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/min_confidence/budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/min_confidence/seed_965/budget_30.pkl --sampling_method min_confidence --neptune_study_name classwise_min_confidence_seed965_ood --training_budget 30

# bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/max_variability/budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/max_variability/seed_965/budget_30.pkl --sampling_method max_variability --neptune_study_name classwise_max_variability_seed965_ood --training_budget 30

# bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/min_variability/budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/min_variability/seed_965/budget_30.pkl --sampling_method min_variability --neptune_study_name classwise_min_variability_seed965_ood --training_budget 30

# bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/tophat/alpha_2_beta_1_budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/tophat/seed_965/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_tophat_seed965_ood --training_budget 30