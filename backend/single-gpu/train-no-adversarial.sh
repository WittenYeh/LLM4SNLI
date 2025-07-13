python3 train.py \
    --gpu_id 1 \
    --model_path ../../models/bert-base-uncased \
    --dataset_name snli \
    --epochs 5 \
    --batch_size 128 \
    --lr 2e-5 \
    --output_dir ../post-trained_model
    