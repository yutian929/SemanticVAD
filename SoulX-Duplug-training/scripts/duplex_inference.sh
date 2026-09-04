export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=path_to_your_env/lib:$LD_LIBRARY_PATH

# python -m debugpy --listen 5678 --wait-for-client
python scripts/duplex_inference.py \
    --config_path config/infer_config.yaml \
    --eval_dir ""
