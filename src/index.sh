CUDA_VISIBLE_DEVICES=7 python indexer.py \
    --data_dir ./database \
    --index_save_dir ./index \
    --peft_model_path ./model/peft \
    --batch_size 16 \
    --use_content_type title \
    --language zh \
    --index_name bsharedrag.index