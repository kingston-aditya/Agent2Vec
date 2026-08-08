import ijson
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import os
import decord
from PIL import Image
import random

import sys
root = os.getcwd()
while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, '.git')): root = os.path.dirname(root)
sys.path.insert(1, root)

from dataloaders.base_dataloader import VideoContrastiveDataset

def count_json_items(file_path: str) -> int:
    print(f"Counting items in {file_path}...")
    total_count = 0
    samples = []
    with open(file_path, "rb") as f:
        # ijson parse events emit ('start_map', None) for every dictionary item in the list
        for item in ijson.items(f, "item", multiple_values=True):
            samples.append(item)
            total_count+=1

            if total_count >= 240000:
                break
    print("total count", total_count)
    return samples


class KineticsStreamDataset(VideoContrastiveDataset):
    def __init__(self, data_path: str, video_dir: str):
        super().__init__(data_path)
        self.json_path = data_path
        self.video_dir = video_dir

    @staticmethod
    def load_dataset(data_path):
        return count_json_items(data_path)

    def __getitem__(self, idx):
        item = self.data[idx]

        caption = item["conversations"] + "\n Represent the action of this video."
        neg_prompt = "Video shows " + item["label"]

        try:
            video_tensor = self.load_video(os.path.join(self.video_dir, item["video_path"]), 16)
        except Exception as e:
            print(f"Error is: {e}")
            new_idx = random.randint(0, len(self.data)-1)
            return self.__getitem__(new_idx)
        
        video_view1 = self._sample_frames(video_tensor, num_frames=16, offset=0)
        
        return {
            "video_pos": video_view1,  # Shape: [16, 3, 224, 224]
            "prompts_neg": neg_prompt,  # Shape: [16, 3, 224, 224]
            "prompts": caption
        }


if __name__ == "__main__":
    json_path = "/work/YamadaU/asarkar/agent2vec_outputs/thinking_texts_kinetics.json"
    video_dir = "/bucket/YamadaU/Datasets/k400/train/"

    dataset = KineticsStreamDataset(json_path, video_dir)
    
    # num_workers=0 runs everything in the main process, avoiding file descriptor issues
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dataset.collate_fn,
        num_workers=4
    )

    for batch_idx, batch in enumerate(dataloader):
        pdb.set_trace()
        break