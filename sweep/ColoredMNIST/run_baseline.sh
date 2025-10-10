dataset=ColoredMNIST
command=$1
data_dir=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms ERM IRM GroupDRO Mixup MLDG CORAL MMD DANN MTL SagNet ARM VREx RSC ANDMask RIDG LFME ASGDRO \
       --data_dir ${data_dir}\
       --command_launcher smart_gpu\
       --fixed_test_envs 2\
       --n_hparams 20\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/extend_baselines"