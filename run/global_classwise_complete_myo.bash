modelseed=$1
gpu=$2
subset=$3


bash run/vqa_from_scratch.bash $gpu LXR111/$subset/min_variability/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/min_variability/seed_388/budget_30.pkl --sampling_method min_variability --neptune_study_name classwise_min_variability_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/max_variability/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/max_variability/seed_388/budget_30.pkl --sampling_method max_variability --neptune_study_name classwise_max_variability_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/min_confidence/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/min_confidence/seed_388/budget_30.pkl --sampling_method min_confidence --neptune_study_name classwise_min_confidence_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/max_confidence/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/max_confidence/seed_388/budget_30.pkl --sampling_method max_confidence --neptune_study_name classwise_max_confidence_subset_${subset} --training_budget 30 --seed $modelseed

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/global_min_variability/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/global_min_variability/seed_388/budget_30.pkl --sampling_method global_min_variability --neptune_study_name global_min_variability_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/global_max_variability/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/global_max_variability/seed_388/budget_30.pkl --sampling_method global_max_variability --neptune_study_name global_max_variability_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/global_min_confidence/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/global_min_confidence/seed_388/budget_30.pkl --sampling_method global_min_confidence --neptune_study_name global_min_confidence_subset_${subset} --training_budget 30 --seed $modelseed
bash run/vqa_from_scratch.bash $gpu LXR111/$subset/global_max_confidence/budget_30_modelseed$modelseed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/global_max_confidence/seed_388/budget_30.pkl --sampling_method global_max_confidence --neptune_study_name global_max_confidence_subset_${subset} --training_budget 30 --seed $modelseed

