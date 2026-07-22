export OUTPUT_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"

accelerate launch /nfshomes/asarkar6/aditya/Agent2Vec/training/train_agent_sft.py \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --max_grad_norm=1.0 \
    --mixed_precision="bf16" \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=5000 \
    --max_train_steps=50000 \
    --checkpointing_steps=10 \
    --resume_from_checkpoint=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR