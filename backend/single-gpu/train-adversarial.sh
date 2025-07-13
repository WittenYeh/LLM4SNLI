python3 train.py \
    --gpu_id 1 \
    --model_path ../models/bert-base-uncased \
    --dataset_name snli \
    --epochs 3 \
    --batch_size 128 \
    --lr 2e-5 \
    --do_adversarial_training \
    --output_dir ../post-trained_model
    