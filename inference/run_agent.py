import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from mmada.training.utils import get_config
from dataloaders.unified_dataloader import UnifiedDataloader
from dataloaders.onevision_dataloader import onevision_dataloader
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader, Subset


def make_inference(videos, questions, answers):
    # create the messages
    messages = []
    for _, (video, question, answer) in enumerate(zip(videos, questions, answers)):
        messages.append([{
            "role":"user",
            "content":[
                {
                    "type": "video",
                    "video": video,
                    "max_frames": 64,
                    "sample_fps": 1.0
                },
                {
                    "type":"text",
                    "text":f"{gen_prompt} Question: {question}. Answer: {answer}."
                }
            ]
        }])

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,) for msg in messages]

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size= 16,
        return_video_metadata=True
    )

    # prepare the metadata
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    # ForkedPdb().set_trace()

    # video_kwargs["do_sample_frames"] = [video_kwargs["do_sample_frames"]]*2

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cuda')

    generated_ids = model.generate(
        **inputs, max_new_tokens=1024, num_beams=1, do_sample=False, temperature=0.0,
    )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text

