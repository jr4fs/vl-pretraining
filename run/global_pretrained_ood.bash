bash run/vqa_ood_pretrained.bash 0 LXR111/animals/full_training_run/ --subset animals --sampling_method none --neptune_study_name full_run_pretrained_ood --training_budget 100
bash run/vqa_ood_pretrained.bash 0 LXR111/animals/global_min_variability/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_min_variability/seed_388/budget_30.pkl --sampling_method global_min_variability --neptune_study_name global_min_variability_seed388_pretrained_ood --training_budget 30
bash run/vqa_ood_pretrained.bash 0 LXR111/animals/global_max_variability/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_max_variability/seed_388/budget_30.pkl --sampling_method global_max_variability --neptune_study_name global_max_variability_seed388_pretrained_ood --training_budget 30
bash run/vqa_ood_pretrained.bash 0 LXR111/animals/global_min_confidence/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_min_confidence/seed_388/budget_30.pkl --sampling_method global_min_confidence --neptune_study_name global_min_confidence_seed388_pretrained_ood --training_budget 30
bash run/vqa_ood_pretrained.bash 0 LXR111/animals/global_max_confidence/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_max_confidence/seed_388/budget_30.pkl --sampling_method global_max_confidence --neptune_study_name global_max_confidence_seed388_pretrained_ood --training_budget 30
bash run/vqa_ood_pretrained.bash 0 LXR111/animals/global_random/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_random/seed_388/budget_30.pkl --sampling_method global_random --neptune_study_name global_random_seed388_pretrained_ood --training_budget 30