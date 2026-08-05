import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from accelerate import Accelerator, DataLoaderConfiguration
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
from dataloaders.activity_dataloader import ActivityNetCaptionsTarDataset

import json
import pandas as pd
import matplotlib.pyplot as plt

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
        
        # Initialize projections safely
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, vjepa_inputs, llava_inputs):
        with torch.no_grad():
            # llava outputs
            llava_outputs = self.llava_model(
                    input_ids=llava_inputs.input_ids,
                    attention_mask=llava_inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            llava_feats = llava_outputs.hidden_states[-1]

            # jepa outputs
            if vjepa_inputs is not None:
                v_jepa_feats = self.vjepa_model(**vjepa_inputs).last_hidden_state
            else:
                v_jepa_feats = None

        projected_llava = self.llava_projector(llava_feats)
        
        # Concatenate tokens sequence-wise
        combined_tokens = torch.cat([v_jepa_feats, projected_llava], dim=1) if v_jepa_feats is not None else projected_llava
        
        # Pass through attention layers
        transformed_tokens = self.transformer_encoder(combined_tokens)

        # STABILITY FIX: Use index 0 (or mean pooling) rather than -1 to avoid padding artifacts
        fused_representations = self.embedding_projection(transformed_tokens[:, 0, :])
        
        return fused_representations

    @staticmethod
    def gather_embeddings_with_grad(tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor

        gathered_tensors = dist_nn.all_gather(tensor)
        return torch.cat(gathered_tensors, dim=0)

    def compute_infonce_loss(self, query_embeds, target_embeds, temp=0.07, max_logit=50.0):
        # 1. Cast to FP32 immediately for numerical stability
        query_32 = query_embeds.float()
        target_32 = target_embeds.float()

        # 2. L2 Normalization in local rank
        q_norm_local = F.normalize(query_32, p=2, dim=-1)
        t_norm_local = F.normalize(target_32, p=2, dim=-1)

        # 3. Gather across distributed GPUs
        q_norm = self.gather_embeddings_with_grad(q_norm_local)
        t_norm = self.gather_embeddings_with_grad(t_norm_local)

        # 4. Compute scaled similarity matrix
        similarity_matrix = torch.matmul(q_norm, t_norm.T) / temp

        # 5. Logit clamping to prevent exponential explosion in Softmax
        similarity_matrix = torch.clamp(similarity_matrix, min=-max_logit, max=max_logit)

        device = query_embeds.device
        
        # STABILITY FIX: Correct label indexing across distributed ranks
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            batch_size = query_embeds.size(0)
            start_idx = rank * batch_size
            end_idx = start_idx + batch_size
            labels = torch.arange(start_idx, end_idx, device=device)
            
            # Extract local rows for cross-entropy to conserve memory & keep loss localized
            local_sim_matrix = similarity_matrix[start_idx:end_idx]
            
            loss_q2t = F.cross_entropy(local_sim_matrix, labels)
            loss_t2q = F.cross_entropy(similarity_matrix.T[start_idx:end_idx], labels)
        else:
            batch_size = query_embeds.size(0)
            labels = torch.arange(batch_size, device=device)
            loss_q2t = F.cross_entropy(similarity_matrix, labels)
            loss_t2q = F.cross_entropy(similarity_matrix.T, labels)

        total_loss = (loss_q2t + loss_t2q) / 2.0
        return total_loss

def plot_json_metric(json_path="training_logs_st2.json", metric_name="loss", smooth_window=20):
    """
    Plots a metric from the JSON log file against iterations.
    
    Available metrics: 'loss', 'loss1', 'loss2', 'lr', 'grad_norm'
    """
    data = []
    
    # Read the JSON lines file
    with open(json_path, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in JSON. Available: {list(df.columns)}")

    plt.figure(figsize=(10, 5))

    # Plot raw values
    plt.plot(df["step"], df[metric_name], alpha=0.3, color="royalblue", label="Raw Step Values")

    # Plot moving average for cleaner trends
    if smooth_window > 1 and len(df) > smooth_window:
        smoothed = df[metric_name].rolling(window=smooth_window).mean()
        plt.plot(df["step"], smoothed, color="navy", linewidth=2, label=f"Smoothed ({smooth_window} steps)")

    plt.xlabel("Iterations (Global Step)", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f"{metric_name.replace('_', ' ').title()} vs. Iterations", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Use logarithmic scale if plotting learning rate
    if metric_name == "lr":
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(f"{metric_name}_vs_iterations.png", dpi=300)
    plt.show()

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
    dataloader_config = DataLoaderConfiguration(even_batches=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        dataloader_config = dataloader_config
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
    tar_parts_dir = "/bucket/YamadaU/asarkar/"
    dataset = ActivityNetCaptionsTarDataset(data_path=data_path, tar_parts_dir=tar_parts_dir)
    # root = "/nfshomes/asarkar6/trinity/small_video_dst/"
    # data_path = {"root": root, "video": os.path.join(root, "videos.txt"), "captions": os.path.join(root, "captions.txt")}
    # dataset = VideoCaptionDataset(data_path=data_path)

    collate_fn = dataset.collate_fn

    # dataset = ConcatDataset([dataset] * int(10e7))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate/(args.gradient_accumulation_steps*accelerator.num_processes*args.train_batch_size), 
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
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def get_latest_checkpoint(checkpoint_dir):
        dirs = [d for d in Path(checkpoint_dir).glob("jean2-checkpoint-*")]
        if not dirs:
            return None
        # Sort by step number (assuming folders named like 'checkpoint-1000')
        latest_dir = sorted(dirs, key=lambda x: int(str(x).split("-")[-1].split(".")[0]))[-1]
        return str(latest_dir)

    initial_global_step = 0

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

            initial_global_step = int(latest_checkpoint.split("-")[-1].split(".")[0])
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
    global_step = initial_global_step
    first_epoch = 0

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
                # get the entities from batch
                videos_query = batch["video_pos"] 
                prompts_query = batch["prompts"]

                prompts_target = batch["prompts_neg"]

                # get inputs from v_jepa
                vjepa_inputs = vjepa_processor(videos_query, return_tensors="pt").to(accelerator.device)

                # get inputs from llava-ov for query
                formatted_prompts = []
                for text_prompt in prompts_query:
                    conversation = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
                    prompt_formatted = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    formatted_prompts.append(prompt_formatted)
                llava_inputs = llava_processor(text=formatted_prompts, padding=True, return_tensors="pt").to(accelerator.device)

                # get inputs from llava-ov for target
                formatted_prompts = []
                for text_prompt in prompts_target:
                    conversation = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
                    prompt_formatted = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    formatted_prompts.append(prompt_formatted)
                llava_inputs = llava_processor(text=formatted_prompts, padding=True, return_tensors="pt").to(accelerator.device)
                
                query_embeds = model(vjepa_inputs, llava_inputs)
                target_embeds = model(None, llava_inputs)

                loss = model.module.compute_infonce_loss(query_embeds, target_embeds)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        model.parameters()
                    )
                    grad_norm_tensor = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    grad_norm = grad_norm_tensor.item() if grad_norm_tensor is not None else 0.0

                    if grad_norm > 100:
                        optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("jean2-checkpoint")]
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
                                    os.remove(removing_checkpoint)

                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, f"jean2-checkpoint-{global_step}.pt"))
                            logger.info("Training cycle complete. Core weights successfully serialized.")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process and global_step % 2 != 0:
                new_logs ={
                    "step": global_step,
                    "loss": float(loss.detach().item()),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm),
                }

                json_file = os.path.join(args.output_dir, "training_logs_st2.json")
                mode = "w" if global_step == 1 else "a"
                with open(json_file, mode=mode, encoding="utf-8") as f:
                    f.write(json.dumps(new_logs) + "\n")

            if global_step >= args.max_train_steps:
                break
    
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)

    plot_json_metric(os.path.join(args.output_dir, "training_logs_st2.json"), metric_name="loss")
    plot_json_metric(os.path.join(args.output_dir, "training_logs_st2.json"), metric_name="grad_norm")
    plot_json_metric(os.path.join(args.output_dir, "training_logs_st2.json"), metric_name="lr")