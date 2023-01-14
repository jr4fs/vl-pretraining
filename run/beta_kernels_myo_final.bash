seed=$1
gpu=$2
kernel=$3
subset=$4

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/beta/beta_kernel/$kernel/alpha_2_beta_2_budget_30_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/beta/beta_kernel/$kernel/seed_$seed/alpha_2_beta_2_budget_30.pkl --sampling_method beta --neptune_study_name beta_22_${kernel}_${subset} --training_budget 30 --seed 965
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/beta/beta_kernel/$kernel/alpha_2_beta_1_budget_30_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/beta/beta_kernel/$kernel/seed_$seed/alpha_2_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_21_${kernel}_${subset} --training_budget 30 --seed 965
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/beta/beta_kernel/$kernel/alpha_1_beta_2_budget_30_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/beta/beta_kernel/$kernel/seed_$seed/alpha_1_beta_2_budget_30.pkl --sampling_method beta --neptune_study_name beta_12_${kernel}_${subset} --training_budget 30 --seed 965
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/beta/beta_kernel/$kernel/alpha_1_beta_1_budget_30_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/beta/beta_kernel/$kernel/seed_$seed/alpha_1_beta_1_budget_30.pkl --sampling_method beta --neptune_study_name beta_11_${kernel}_${subset} --training_budget 30 --seed 965

