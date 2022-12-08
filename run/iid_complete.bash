bash run/vqa_from_scratch.bash 2 LXR111/animals/beta/beta_pvals/alpha_2_beta_1_budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_pvals/seed_388/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_pvals_seed388 --training_budget 30
bash run/vqa_from_scratch.bash 2 LXR111/animals/beta/beta_kernel/gaussian/alpha_1_beta_2_budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/gaussian/seed_388/alpha_1_beta_2_budget_30.pkl --sampling_method beta --neptune_study_name beta_12_gaussian_seed388 --training_budget 30
bash run/vqa_from_scratch.bash 2 LXR111/animals/beta/beta_kernel/exponential/alpha_2_beta_1_budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/exponential/seed_388/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_exponential_seed388 --training_budget 30
bash run/vqa_from_scratch.bash 2 LXR111/animals/beta/beta_kernel/linear/alpha_1_beta_1_budget_30_seed388/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/beta/beta_kernel/linear/seed_388/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_linear_seed388 --training_budget 30
