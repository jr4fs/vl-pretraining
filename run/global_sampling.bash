seed=$1
algorithm=$2 
gpu=$3

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_10_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_10.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed --training_budget 10

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_20_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_20.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed --training_budget 20

bash run/vqa_from_scratch.bash $gpu LXR111/animals/$algorithm/budget_30_seed$seed/ --subset animals --sampling_ids src/dataset_selection/sampling/samples/LXR111/animals/$algorithm/seed_$seed/budget_30.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_seed$seed --training_budget 30
