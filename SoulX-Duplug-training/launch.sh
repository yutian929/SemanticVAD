# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_LIBRARY_PATH=path_to_your_env/lib:$LD_LIBRARY_PATH
# export WANDB_API_KEY=

wandb login


cmd="torchrun finetune.py --config_path config/train_config.yaml"
# cmd="python finetune.py --config_path config/train_config.yaml"

echo $cmd
eval $cmd


# debug_cmd="python -m debugpy --listen 5678 --wait-for-client finetune.py \
# --config_path config/debug_config.yaml"

# echo $debug_cmd
# eval $debug_cmd
