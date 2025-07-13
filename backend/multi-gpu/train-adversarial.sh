TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=8 dist_train.py \
    --model_path ../models/bert-base-uncased \
    --dataset_name snli \
    --epochs 5 \
    --batch_size 256 \
    --lr 2e-5 \
    --do_adversarial_training \
    --output_dir ../../post-trained_model/
    