import os
import json
import io
import tarfile
import numpy as np
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from decord import VideoReader, cpu
from dataloaders.base_dataloader import VideoContrastiveDataset

import pdb

class ActivityNetCaptionsTarDataset(VideoContrastiveDataset):
    def __init__(self, data_path: str, tar_parts_dir: str):
        """
        Args:
            data_path: Path to activitynet_captions_train.json
            tar_parts_dir: Folder containing ActivityNet_Videos.tar.part-000...007
        """
        self.tar_parts_dir = tar_parts_dir
        
        # 1. Gather sorted list of part files
        self.part_paths = sorted([
            os.path.join(tar_parts_dir, f) for f in os.listdir(tar_parts_dir)
            if "ActivityNet_Videos.tar.part-" in f
        ])
        
        # Calculate cumulative file byte offsets once
        self.part_sizes = [os.path.getsize(p) for p in self.part_paths]
        self.part_offsets = np.cumsum([0] + self.part_sizes[:-1])
        
        # 2. Build lightweight byte index (takes ~2-3s)
        self.tar_index = self._build_fast_tar_index()
        
        # 3. Call parent dataset initialization
        super().__init__(data_path)

    def _build_fast_tar_index(self):
        """Scans TAR headers once to map video filenames to global byte offsets."""
        index = {}
        
        class CombinedStream(io.RawIOBase):
            def __init__(self, paths):
                self.paths = paths
                self.f = None
                self.idx = 0
                self._open_next()

            def _open_next(self):
                if self.f:
                    self.f.close()
                if self.idx < len(self.paths):
                    self.f = open(self.paths[self.idx], "rb")
                    self.idx += 1

            def readinto(self, b):
                if not self.f:
                    return 0
                n = self.f.readinto(b)
                if n == 0 and self.idx < len(self.paths):
                    self._open_next()
                    return self.readinto(b)
                return n

        with tarfile.open(fileobj=CombinedStream(self.part_paths), mode="r|*") as tar:
            for member in tar:
                if member.isfile():
                    filename = os.path.basename(member.name)
                    index[filename] = {
                        "offset": member.offset_data,
                        "size": member.size
                    }
        return index

    def load_dataset(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        records = []
        for video_id, item in raw_data.items():
            sentences = item.get("sentences", [])
            full_caption = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            records.append({
                "video_id": video_id,
                "video_filename": f"{video_id}.mp4",
                "caption": full_caption
            })

        return HFDataset.from_list(records)

    def _read_bytes_at_offset(self, global_offset: int, size: int) -> bytes:
        """Direct seek and read from specific .part files."""
        part_idx = np.searchsorted(self.part_offsets, global_offset, side="right") - 1
        local_offset = global_offset - self.part_offsets[part_idx]
        
        data = bytearray()
        remaining = size
        curr_part = part_idx
        curr_local_offset = local_offset
        
        while remaining > 0 and curr_part < len(self.part_paths):
            with open(self.part_paths[curr_part], "rb") as f:
                f.seek(curr_local_offset)
                chunk = f.read(min(remaining, self.part_sizes[curr_part] - curr_local_offset))
                data.extend(chunk)
                remaining -= len(chunk)
            curr_part += 1
            curr_local_offset = 0
            
        return bytes(data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        filename = item["video_filename"]

        if filename not in self.tar_index:
            raise FileNotFoundError(f"Video {filename} not in TAR index.")

        meta = self.tar_index[filename]
        video_bytes = self._read_bytes_at_offset(meta["offset"], meta["size"])

        buf = io.BytesIO(video_bytes)
        vr = VideoReader(buf, ctx=cpu(0), width=224, height=224)
        
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Corrupted video: {filename}")

        indices = np.linspace(0, total_frames - 1, num=32, dtype=int).tolist()
        video_tensor = vr.get_batch(indices).asnumpy()

        video_view1 = self._sample_frames(video_tensor, num_frames=16, offset=0)
        video_view2 = self._sample_frames(video_tensor, num_frames=16, offset=8)

        return {
            "video_pos": video_view1,
            "video_neg": video_view2,
            "prompts": item["caption"]
        }

if __name__ == "__main__":
    DATA_DIR = ""
    dataset = ActivityNetCaptionsTarDataset(data_path = os.path.join(DATA_DIR, ), tar_parts_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
    
    for i, batch in enumerate(dataloader):
        pdb.set_trace()
        break

