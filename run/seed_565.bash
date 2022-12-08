bash run/vqa_from_scratch.bash 0 LXR111/animals/full_run/modelseed_565/ --subset animals --sampling_method none --neptune_study_name full_run_modelseed565 --training_budget 100

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/full_run/modelseed_565/ --subset animals --sampling_method none --neptune_study_name full_run_modelseed565_ood --training_budget 100

bash run/global_classwise_ood_complete.bash 565 0