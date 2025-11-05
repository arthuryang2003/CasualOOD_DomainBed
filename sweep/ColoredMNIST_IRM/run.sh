dataset=ColoredMNIST_IRM
command=$1
data_dir=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms VITA\
       --data_dir ${data_dir}\
       --command_launcher smart_gpu\
       --fixed_test_envs 2\
       --n_hparams 10\
       --n_trials 1\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/VITA_v2_test6"

