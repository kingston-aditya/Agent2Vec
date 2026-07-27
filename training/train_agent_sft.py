import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs

import shutil
from pathlib import Path

from transformers import (
    AutoProcessor,  
    AutoVideoProcessor,          
    AutoModel,               
    LlavaOnevisionForConditionalGeneration, 
    get_scheduler       
)
import logging
import os
import math
from tqdm import tqdm
import argparse

logger = get_logger(__name__)

import sys
import pdb as pdb_original

root = os.getcwd()
while not os.path.isdir(os.path.join(root, '.git')): root = os.path.dirname(root)
sys.path.insert(1, root)

from dataloaders.test_dataloader import VideoCaptionDataset
from dataloaders.cc3m_video_dataloader import CC3MPatchHFDataset

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class JointEmbeddingAlignmentNetwork(nn.Module):
    def __init__(self, vjepa_model, llava_model, v_jepa_dim=1024, llava_dim=896, num_layers=4, nhead=8, dim_feedforward=2048):
        super().__init__()

        self.vjepa_model = vjepa_model
        self.llava_model = llava_model

        self.embedding_projection = nn.Sequential(
            nn.Linear(v_jepa_dim, v_jepa_dim),
            nn.GELU(),
            nn.Linear(v_jepa_dim, 768)  
        )
        
        # 2-Layer MLP Projection for LLaVA-OV (896 -> 1024)
        self.llava_projector = nn.Sequential(
            nn.Linear(llava_dim, v_jepa_dim),
            nn.GELU(),
            nn.Linear(v_jepa_dim, v_jepa_dim)
        )
        
        # N Self-Attention Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=v_jepa_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable temperature for standard InfoNCE scaling
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, vjepa_inputs, llava_inputs):
        with torch.no_grad():
            # llava outputs
            llava_outputs = self.llava_model(
                    input_ids=llava_inputs.input_ids,
                    attention_mask=llava_inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            llava_feats = llava_outputs.hidden_states[-1].float()

            # jepa outputs
            v_jepa_feats = self.vjepa_model(**vjepa_inputs).last_hidden_state

        projected_llava = self.llava_projector(llava_feats)
        
        # Concatenate tokens sequence-wise: [CLS] + Video + Language
        combined_tokens = torch.cat([v_jepa_feats, projected_llava], dim=1)
        
        # Pass through attention layers
        transformed_tokens = self.transformer_encoder(combined_tokens)

        fused_representations = self.embedding_projection(transformed_tokens[:, -1, :])
        
        # Extract the processed CLS representation [B, 1024]
        return fused_representations

    def compute_infonce_loss(self, cls_representations, temp=0.07):
        total_samples = cls_representations.size(0)
        batch_size = total_samples // 2

        queries = cls_representations  # [2*B, D]
        targets = cls_representations[:batch_size]  # [B, D]

        queries_norm = F.normalize(queries, p=2, dim=-1)
        targets_norm = F.normalize(targets, p=2, dim=-1)

        # Similarity matrix: [2*B, B]
        similarity_matrix = torch.matmul(queries_norm, targets_norm.T) / temp

        device = cls_representations.device
        labels = torch.arange(batch_size, device=device)

        # Row loss uses the positive pairs in the upper half [B, B]
        loss_row = F.cross_entropy(similarity_matrix[:batch_size, :], labels)
        
        # FIX: Slice the columns to align predictions cleanly [B, B]
        loss_col = F.cross_entropy(similarity_matrix[:batch_size, :].T, labels)
        
        total_loss = (loss_row + loss_col) / 2.0
        return total_loss

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # model checkpoints
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the seedraw checkpoint to load.")
    parser.add_argument("--output_dir", type=str, default="sd3-dreambooth", help="Directory for model predictions and checkpoints.")

    # dataset 
    parser.add_argument("--cache_dir", type=str, default="/nfshomes/asarkar6/trinity/model_weights/", help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Sampling batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.")
    parser.add_argument("--checkpointing_steps", type=int, default=100, help="Save a checkpoint of the training state every X updates.")

    # others
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, help="Max number of checkpoints to store.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate before backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")

    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Scheduler type for learning rate.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses for data loading.")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"], help="Scheme for weighting.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit_normal weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Standard deviation for logit_normal weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting scheme.")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay for UNet parameters.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam/Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report logs to; can be 'tensorboard', 'wandb', or 'comet_ml'.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mode for mixed precision training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")
    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Total number of processing nodes/GPUs detected: {accelerator.num_processes}")
    
    # distribute Jepa on different devices
    vjepa_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16, cache_dir=args.cache_dir)
    
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    )
    logger.info("loaded Vjepa and LLava")

    # distribute model on different devices
    model = JointEmbeddingAlignmentNetwork(vjepa_model, llava_model, v_jepa_dim=vjepa_model.config.hidden_size, llava_dim=llava_model.config.text_config.hidden_size, num_layers=4)
    model.to(accelerator.device)

    model.requires_grad_(True)
    model.vjepa_model.requires_grad_(False)
    model.llava_model.requires_grad_(False)

    # load the dataset
    data_path = "/bucket/YamadaU/asarkar/CC3M/"
    dataset = CC3MPatchHFDataset(data_path=data_path)
    collate_fn = dataset.collate_fn
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    estimated_dataset_size = len(dataloader)
    num_update_steps_per_epoch = math.ceil(estimated_dataset_size / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    def get_latest_checkpoint(checkpoint_dir):
        dirs = [d for d in Path(checkpoint_dir).glob("jean-checkpoint-*")]
        if not dirs:
            return None
        # Sort by step number (assuming folders named like 'checkpoint-1000')
        latest_dir = sorted(dirs, key=lambda x: int(str(x).split("-")[-1].split(".")[0]))[-1]
        return str(latest_dir)

    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        latest_checkpoint = get_latest_checkpoint(args.resume_from_checkpoint)
        if latest_checkpoint:
            accelerator.print(f"Loading checkpoint from: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            accelerator.print("No checkpoint found. Starting training from scratch.")


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    model.train()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataloader)}")
    logger.info(f"  Num batches each epoch = {total_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # get the entities from batch
                videos = batch["video_pos"] + batch["video_neg"]
                prompts = batch["prompts"] + [""]*len(batch["video_neg"])

                # get inputs from v_jepa
                vjepa_inputs = vjepa_processor(videos, return_tensors="pt").to(accelerator.device)

                # get inputs from llava-ov
                formatted_prompts = []
                for text_prompt in prompts:
                    conversation = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
                    prompt_formatted = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    formatted_prompts.append(prompt_formatted)
                llava_inputs = llava_processor(text=formatted_prompts, padding=True, return_tensors="pt").to(accelerator.device)
                

                cls_representations = model(vjepa_inputs, llava_inputs)
                loss = model.module.compute_infonce_loss(cls_representations)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        model.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("connector-checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, f"jean-checkpoint-{global_step}.pt"))
                        logger.info("Training cycle complete. Core weights successfully serialized.")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)