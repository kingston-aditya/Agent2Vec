import ijson
from torch.utils.data import IterableDataset, DataLoader
import pdb
import torch
import os
import decord
from PIL import Image
import random

def sample_frame_indices(total_frames: int, num_frames: int = 8) -> list:
    """Uniformly sample frame indices across total video length."""
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    
    interval = total_frames / float(num_frames)
    return [int(i * interval) for i in range(num_frames)]

def count_json_items(file_path: str) -> int:
    print(f"Counting items in {file_path}...")
    total_count = 0
    with open(file_path, "rb") as f:
        # ijson parse events emit ('start_map', None) for every dictionary item in the list
        for item in ijson.items(f, "item", multiple_values=True):
            total_count+=1
    print("total count", total_count)
    return samples

def load_video_frames(video_path: str, num_frames: int = 16) -> torch.Tensor:
    vr = decord.VideoReader(
        video_path, 
        ctx=decord.cpu(0), 
        width=224, 
        height=224, 
        num_threads=1
    )
    
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError("Decord returned 0 total frames.")

    # Get sampled frame indices
    indices = sample_frame_indices(total_frames, num_frames=num_frames)
    
    # Extract batch of frames -> Returns Tensor of shape [16, H, W, C]
    video_tensor = vr.get_batch(indices).asnumpy()
    pil_frames_list = [Image.fromarray(frame) for frame in video_tensor]
    return pil_frames_list

class KineticsStreamDataset(IterableDataset):
    def __init__(self, json_path: str, video_dir: str):
        super().__init__()
        self.json_path = json_path
        self.video_dir = video_dir

    def __iter__(self):
        with open(self.json_path, "rb") as f:
            for record in ijson.items(f, "item", multiple_values=True):
                record["video_pos"] = load_video_frames(os.path.join(self.video_dir, record["video_path"]), num_frames=8)
                record["prompts_neg"] = "Video shows " + record["label"]
                record["prompts"] = record["conversations"] + "\n Represent the action in this video."
                yield record

    def __len__(self):
        return count_json_items(self.json_path)

    @staticmethod
    def custom_collate_fn(batch):
        return {
            "video_pos": [item["video_pos"] for item in batch],
            "prompts": [item["prompts"] for item in batch],
            "prompts_neg": [item["prompts_neg"] for item in batch],
        }


def get_single_process_dataloader(json_path: str, video_dir: str, batch_size: int = 16):
    dataset = KineticsStreamDataset(json_path, video_dir)
    
    # num_workers=0 runs everything in the main process, avoiding file descriptor issues
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.custom_collate_fn,
        num_workers=4
    )
    return dataloader


if __name__ == "__main__":
    json_path = "/work/YamadaU/asarkar/agent2vec_outputs/thinking_texts_kinetics.json"
    video_dir = "/bucket/YamadaU/Datasets/k400/train/"

    dataloader = get_single_process_dataloader(json_path, video_dir, batch_size=16)

    print("Streaming dataset...")
    for batch_idx, batch in enumerate(dataloader):
        pdb.set_trace()
        break