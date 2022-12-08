bash run/vqa_from_scratch.bash 0 LXR111/animals/full_run/modelseed_388/ --subset animals --sampling_method none --neptune_study_name full_run_modelseed388 --training_budget 100

bash run/vqa_ood_from_scratch.bash 0 LXR111/animals/full_run/modelseed_388/ --subset animals --sampling_method none --neptune_study_name full_run_modelseed388_ood --training_budget 100

bash run/global_classwise_ood_complete.bash 388 0