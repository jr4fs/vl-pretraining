# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

dataset=$1

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/cosine
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/cosine/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/cosine/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/cosine/seed_965

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/epanechnikov
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/epanechnikov/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/epanechnikov/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/epanechnikov/seed_965

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/exponential
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/exponential/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/exponential/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/exponential/seed_965

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/gaussian
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/gaussian/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/gaussian/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/gaussian/seed_965

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/linear
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/linear/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/linear/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/linear/seed_965

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/tophat
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/tophat/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/tophat/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_kernel/tophat/seed_965



mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_pvals
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_pvals/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_pvals/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_pvals/seed_965


mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_var_counts
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_var_counts/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_var_counts/seed_565
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/beta/beta_var_counts/seed_965


mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_confidence
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_confidence/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_confidence/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_confidence/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_confidence 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_confidence/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_confidence/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_confidence/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_variability 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_variability/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_variability/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_max_variability/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_variability
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_variability/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_variability/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_min_variability/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_random
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_random/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_random/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/global_random/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_confidence 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_confidence/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_confidence/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_confidence/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_confidence 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_confidence/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_confidence/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_confidence/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_variability 
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_variability/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_variability/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/max_variability/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_variability
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_variability/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_variability/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/min_variability/seed_565

mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/random
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/random/seed_388
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/random/seed_965
mkdir src/dataset_selection/sampling/samples/LXR111/$dataset/random/seed_565
