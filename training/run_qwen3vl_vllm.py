import os

import torch
from vllm import LLM, SamplingParams

import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import json

from transformers import AutoProcessor

import sys
root = os.getcwd()
while not os.path.isdir(os.path.join(root, '.git')): root = os.path.dirname(root)
sys.path.insert(1, root)

from dataloaders.test_dataloader import VideoCaptionDataset
from dataloaders.kinetics_dataloader import KineticsCSVDataset, kinetics_collate_fn

from qwen_vl_utils import process_vision_info
from PIL import Image

import pdb
import re

os.environ["HF_HOME"] = "/work/YamadaU/asarkar/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_P2P_DISABLE"] = "1"


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def make_inference(video_frames, answer):
    messages = [
        [{
            "role": "user",
            "content": [
              {
                  "type": "video",
                  "video": video,
                  "max_frames": 8,
                  "sample_fps": 1.0,
              },
              {"type": "text", "text": f"Describe your thinking process while captioning this video as {item}."},
            ],
        }] for video, item in zip(video_frames, answer)
    ]

    inputs = [prepare_inputs_for_vllm(message, processor) for message in messages]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    temp_outs = []
    for o in outputs:
        temp_outs.append(o.outputs[0].text)

    return temp_outs

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1},
        max_model_len=16864,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        mm_encoder_tp_mode="data",
        max_num_seqs=16,
        max_num_batched_tokens=32768,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        padding_side="left",
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, top_p=0.9)

    dataset = KineticsCSVDataset(
        csv_path="/bucket/YamadaU/Datasets/k400/annotations/train.csv",
        video_dir="/bucket/YamadaU/Datasets/k400/train/",
        num_frames=8
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=kinetics_collate_fn,
        pin_memory=True
    )

    data_path = "/work/YamadaU/asarkar/agent2vec_outputs/"

    json_item = []
    for batch in dataloader:
        captions = batch["label_str"]
        video_frames = batch["video"]
        video_path = batch["filename"]

        thinking_texts = make_inference(video_frames, captions)
        
        for idx in range(len(captions)):
            json_item.append({"label": batch["label_str"][idx], "video_path": batch["filename"][idx], "conversations":  thinking_texts[idx]})

        # create a json file in output directory
        with open(os.path.join(data_path, "thinking_texts_kinetics.json"), "a", encoding="utf-8") as f:
            json.dump(json_item, f, indent=2, ensure_ascii=False)