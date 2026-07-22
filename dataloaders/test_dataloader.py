import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataloaders.base_dataloader import VideoContrastiveDataset
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

import pdb

class VideoCaptionDataset(VideoContrastiveDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

    @staticmethod
    def load_dataset(data_path):
        with open(data_path["video"], "r", encoding="utf-8") as f:
            video_paths = [line.strip() for line in f]

        with open(data_path["captions"], "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f]

        assert len(video_paths) == len(captions), \
            "Number of videos and captions must match."
        
        return [{"video": os.path.join(data_path["root"], video_path), "caption": caption} for (video_path, caption) in zip(video_paths, captions)]
    
if __name__ == "__main__":
    root = "/nfshomes/asarkar6/trinity/small_video_dst/"
    data_path = {
        "root": root,
        "video": os.path.join(root, "videos.txt"),
        "captions": os.path.join(root, "captions.txt") 
    }
    
    dataset = VideoCaptionDataset(data_path=data_path)
    collate_fn = dataset.collate_fn

    dataset = ConcatDataset([dataset] * int(10e3))

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        pdb.set_trace()

        print(batch["video"])      
        print(batch["caption"])  
        break