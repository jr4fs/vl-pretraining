seed=$1
algorithm=$2 
gpu=$3
subset=$4

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_10_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_10.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget10 --training_budget 10 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_20_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_20.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget20 --training_budget 20 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_30_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_30.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget30 --training_budget 30 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_40_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_40.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget40 --training_budget 40 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_50_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_50.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget50 --training_budget 50 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_60_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_60.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget60 --training_budget 60 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_70_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_70.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget70 --training_budget 70 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_80_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_80.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget80 --training_budget 80 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_90_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_90.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget90 --training_budget 90 --seed 965

bash run/vqa_from_scratch.bash $gpu LXR111/$subset/$algorithm/budget_100_seed$seed/ --subset $subset --sampling_ids src/dataset_selection/sampling/samples/LXR111/$subset/$algorithm/seed_$seed/budget_100.pkl --sampling_method $algorithm --neptune_study_name ${algorithm}_subset_${subset}_budget100 --training_budget 100 --seed 965

