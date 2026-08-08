export OUTPUT_DIR="/work/YamadaU/asarkar/agent2vec_outputs/"
export CACHE_DIR="/work/YamadaU/asarkar/agent2vec_outputs/weights"

accelerate launch /work/YamadaU/asarkar/Agent2Vec/training/train_agent_sft_stage2.py \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --num_train_epochs=500 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --max_grad_norm=1.0 \
    --mixed_precision="bf16" \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --max_train_steps=10000 \
    --checkpointing_steps=1000 \
    --resume_from_checkpoint=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR
