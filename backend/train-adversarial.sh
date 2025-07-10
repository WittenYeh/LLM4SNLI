python3 train.py \
    --model_path ~/LLM4SNLI/models/bert-base-uncased/ \
    --dataset_name snli \
    --epochs 10 \
    --batch_size 64 \
    --do_adversarial_training \
    --output_dir ../post-trained-model/bert-snli-adversarial/ \
    --gpu_id 1