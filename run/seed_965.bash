bash run/vqa_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/gaussian/alpha_1_beta_1_budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/gaussian/seed_965/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_gaussian_seed965 --training_budget 30


bash run/vqa_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/gaussian/alpha_1_beta_1_budget_30_seed565/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/gaussian/seed_565/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_gaussian_seed565 --training_budget 30


bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/gaussian/alpha_1_beta_1_budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/gaussian/seed_965/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_gaussian_seed965_ood --training_budget 30

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/gaussian/alpha_1_beta_1_budget_30_seed565/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/gaussian/seed_565/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_gaussian_seed565_ood --training_budget 30


bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/tophat/alpha_2_beta_1_budget_30_seed965/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/tophat/seed_965/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_tophat_seed965_ood --training_budget 30

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/beta/beta_kernel/tophat/alpha_2_beta_1_budget_30_seed565/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/tophat/seed_565/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_tophat_seed565_ood --training_budget 30



bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/global_random/budget_30_seed565/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_random/seed_565/budget_30.pkl --sampling_method global_random --neptune_study_name global_random_seed565_ood --training_budget 30

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/global_random/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/global_random/seed_388/budget_30.pkl --sampling_method global_random --neptune_study_name global_random_seed388_ood --training_budget 30



bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/random/budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/random/seed_388/budget_30.pkl --sampling_method random --neptune_study_name classwise_random_seed388_ood --training_budget 30

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/random/budget_30_seed565/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/random/seed_565/budget_30.pkl --sampling_method random --neptune_study_name classwise_random_seed565_ood --training_budget 30
