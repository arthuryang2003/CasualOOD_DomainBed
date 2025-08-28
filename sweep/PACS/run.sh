dataset=PACS
command=$1
data_dir=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms VITA_Zu_only VITA \
       --data_dir ${data_dir}\
       --command_launcher smart_gpu\
       --single_test_envs \
       --n_hparams 2\
       --n_trials 2\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/VITA_wo_Adapt"

