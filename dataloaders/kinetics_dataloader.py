import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import random
import pdb

from PIL import Image

import decord


def sample_frame_indices(total_frames: int, num_frames: int = 8) -> list:
    """Uniformly sample frame indices across total video length."""
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    
    interval = total_frames / float(num_frames)
    return [int(i * interval) for i in range(num_frames)]


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

def load_corrupted_list(file_path: str = "corrupted_videos.json") -> list:
    """Loads the corrupted video paths list from the saved file."""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("corrupted_video_paths", [])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file format. Use .json or .txt.")

class KineticsCSVDataset(Dataset):
    """
    Dataset that constructs Kinetics video filenames from train.csv metadata:
    {youtube_id}_{start:06d}_{end:06d}.mp4
    """
    def __init__(self, csv_path: str, video_dir: str, num_frames: int = 16, transform=None):
        """
        Args:
            csv_path (str): Path to train.csv.
            video_dir (str): Folder containing the downloaded .mp4 video files.
            num_frames (int): Number of frames to sample per video.
            transform (callable, optional): Torchvision transform.
        """
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

        self.corrupt_list = list(load_corrupted_list("./../training/corrupted_videos.txt"))

        # 1. Read train.csv
        df = pd.read_csv(self.csv_path)

        # 2. Map distinct text labels to class indices
        unique_labels = sorted(df['label'].unique().tolist())
        self.label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self.id_to_label = {idx: lbl for lbl, idx in self.label_to_id.items()}

        # 3. Construct expected filename for each CSV entry
        # Pattern: {youtube_id}_{time_start:06d}_{time_end:06d}.mp4
        df['filename'] = df.apply(
            lambda row: f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4", 
            axis=1
        )

        # 4. Scan disk to match available video files
        existing_files = set(os.listdir(self.video_dir)) if os.path.exists(self.video_dir) else set()
        
        # Filter records to only include videos actually present on disk
        self.samples = []
        missing_count = 0

        for _, row in df.iterrows():
            fname = row['filename']
            if fname in existing_files:
                video_path = os.path.join(self.video_dir, fname)
                if video_path in self.corrupt_list:
                    missing_count += 1
                    continue
                else:
                    self.samples.append({
                        "video_path": video_path,
                        "filename": fname,
                        "youtube_id": row['youtube_id'],
                        "label_str": row['label'],
                        "label_id": self.label_to_id[row['label']]
                    })
            else:
                missing_count += 1

        print(f"Loaded {len(self.samples)} valid videos from {csv_path}.")
        if missing_count > 0:
            print(f"Note: {missing_count} entries in CSV were not found in '{video_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = item["video_path"]

        video_tensor = load_video_frames(video_path, num_frames=self.num_frames)

        return {
            "video": video_tensor,                             # Tensor: [num_frames, 3, H, W]
            "label_id": item["label_id"],                       # Integer class index
            "label_str": item["label_str"],                     # String action label (e.g., 'abseiling')
            "youtube_id": item["youtube_id"],                   # YouTube video ID
            "filename": item["filename"]                        # Full MP4 filename
        }


def kinetics_collate_fn(batch):
    videos = [item["video"] for item in batch]
    label_ids = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    label_strs = [item["label_str"] for item in batch]
    youtube_ids = [item["youtube_id"] for item in batch]
    filenames = [item["filename"] for item in batch]

    return {
        "video": videos,          
        "label_id": label_ids,     
        "label_str": label_strs,   
        "youtube_id": youtube_ids, 
        "filename": filenames      
    }


if __name__ == "__main__":
    CSV_PATH = "/bucket/YamadaU/Datasets/k400/annotations/train.csv"
    VIDEO_DIR = "/bucket/YamadaU/Datasets/k400/train/"  # Directory containing the downloaded .mp4 files

    dataset = KineticsCSVDataset(
        csv_path=CSV_PATH,
        video_dir=VIDEO_DIR,
        num_frames=16
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=kinetics_collate_fn
    )

    for i, batch in enumerate(dataloader):
        pdb.set_trace()