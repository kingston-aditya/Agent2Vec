import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from decord import VideoReader, cpu
import random

class VideoContrastiveDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_dataset(data_path)

    @staticmethod
    def load_dataset(data_path):
        dataset = load_dataset('json', data_files=data_path, split='train')
        return dataset

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_video(video_path, num_frames=32):
        vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
        total_frames = len(vr)
        
        if total_frames <= 0:
            raise ValueError("Video file is empty or corrupted.")
            
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()
        decord_frames = vr.get_batch(indices).asnumpy()
        
        return decord_frames

    def _sample_frames(self, video_tensor, num_frames=16, offset=0):
        total_frames = video_tensor.shape[0]
        
        start_idx = offset
        available_frames = total_frames - start_idx
        
        if available_frames < num_frames:
            # Fallback uniform sampling over whatever frames exist
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return video_tensor[indices]
            
        # Standard uniform sampling across the remaining window
        indices = np.linspace(start_idx, total_frames - 1 - offset, num_frames, dtype=int)
        return video_tensor[indices]

    @staticmethod
    def __choose_one(num):
        caps_list = [
            "Caption this image.",
            "Provide a brief caption describing this video.",
            "Write a clear, concise caption for the given video.",
            "Give this video a suitable caption.",
            "Summarize the visual content of this video in one sentence.",
            "Generate a standard descriptive caption for this image."
        ] 
        return caps_list[num]

    def __getitem__(self, idx):
        item = self.data[idx]

        caption = self.__choose_one(random.randint(0, 5))
        neg_prompt = item["caption"]
        video_tensor = self.load_video(item["video"])
        
        video_view1 = self._sample_frames(video_tensor, num_frames=16, offset=0)
        
        return {
            "video_pos": video_view1,  # Shape: [16, 3, 224, 224]
            "prompts_neg": neg_prompt,  # Shape: [16, 3, 224, 224]
            "prompts": caption
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "video_pos": [x["video_pos"] for x in batch],
            "prompts": [x["prompts"] for x in batch],
            "prompts_neg": [x["prompts_neg"] for x in batch]
        }
