import os
import glob
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    import cv2
    HAS_DECORD = False


def check_video_file(video_path: str) -> tuple[str, bool, str]:
    """
    Attempts to read the video file header.
    Returns: (video_path, is_corrupted, error_message)
    """
    try:
        if HAS_DECORD:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            if len(vr) == 0:
                return video_path, True, "Zero frames reported"
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return video_path, True, "Could not open video capture"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                return video_path, True, "Zero frames reported"
                
        return video_path, False, "OK"

    except Exception as e:
        err_msg = str(e)
        return video_path, True, err_msg


def scan_and_save_corrupted_videos(
    video_dir: str, 
    output_json: str = "corrupted_videos.json",
    num_workers: int = 8, 
    recursive: bool = True
):
    """
    Scans video_dir for all .mp4 files, checks for corruptions, and saves results to disk.
    """
    search_pattern = os.path.join(video_dir, "**", "*.mp4") if recursive else os.path.join(video_dir, "*.mp4")
    video_files = glob.glob(search_pattern, recursive=recursive)

    print(f"Found {len(video_files)} .mp4 file(s) in '{video_dir}'. Starting scan...")

    corrupted_paths = []
    moov_atom_paths = []
    detailed_records = []

    # Run inspection in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_video_file, filepath): filepath for filepath in video_files}
        
        for future in tqdm(as_completed(futures), total=len(video_files), desc="Checking videos"):
            filepath, is_corrupted, err_msg = future.result()
            
            if is_corrupted:
                corrupted_paths.append(filepath)
                detailed_records.append({
                    "video_path": filepath,
                    "error": err_msg
                })
                if "moov" in err_msg.lower() or "atom" in err_msg.lower():
                    moov_atom_paths.append(filepath)

    # 1. Save list to JSON file
    save_payload = {
        "total_scanned": len(video_files),
        "total_corrupted": len(corrupted_paths),
        "total_moov_errors": len(moov_atom_paths),
        "corrupted_video_paths": corrupted_paths,
        "moov_error_paths": moov_atom_paths,
        "details": detailed_records
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    # 2. Save simple text file (one path per line for quick inspect/grep)
    txt_output = output_json.rsplit(".", 1)[0] + ".txt"
    with open(txt_output, "w", encoding="utf-8") as f:
        for p in corrupted_paths:
            f.write(f"{p}\n")

    print("\n" + "=" * 60)
    print("SCAN SUMMARY")
    print("=" * 60)
    print(f"Total .mp4 files scanned : {len(video_files)}")
    print(f"Total corrupted files    : {len(corrupted_paths)}")
    print(f"Explicit 'moov' errors   : {len(moov_atom_paths)}")
    print(f"Saved JSON log to        : {output_json}")
    print(f"Saved plain text list to : {txt_output}")
    print("=" * 60)

    return corrupted_paths


# ==========================================
# Helper Function to Load Saved List Later
# ==========================================
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


# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    TARGET_VIDEO_DIR = "/bucket/YamadaU/Datasets/k400/train/"
    JSON_FILE = "corrupted_videos.json"

    # Step 1: Run scan and save list
    scan_and_save_corrupted_videos(
        video_dir=TARGET_VIDEO_DIR,
        output_json=JSON_FILE,
        num_workers=8
    )

    # Step 2: Test reloading the saved list back into Python
    bad_videos_list = load_corrupted_list(JSON_FILE)
    print(f"\nSuccessfully reloaded list! Total bad videos loaded: {len(bad_videos_list)}")