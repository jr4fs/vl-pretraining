seed=$1
algorithm=$2 
gpu=$3

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_10_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_10.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_20_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_20.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_30_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_30.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_40_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_40.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_50_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_50.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_60_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_60.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_70_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_70.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_80_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_80.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_90_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_90.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_100_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_100.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed

