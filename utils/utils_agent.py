import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModel
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

def process_video_pipeline(video_path, text_query, cache_dir=""):
    # =====================================================================
    # STEP 1: Load video, sample at 1 fps, and sample 256 frames from that
    # =====================================================================
    print("Step 1: Extracting frames...")
    # Initialize decord VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    
    # 1a. Sample at 1 fps
    # Calculate step size to get 1 frame per second
    step_1fps = max(1, round(fps))
    indices_1fps = np.arange(0, len(vr), step_1fps)
    
    # Extract 1 fps frames (Shape: [N, H, W, 3])
    frames_1fps = vr.get_batch(indices_1fps).asnumpy()
    total_1fps_frames = len(indices_1fps)
    
    # 1b. Sample 256 frames uniformly from the 1 fps frames
    # np.linspace ensures we span the entire video. 
    idx_256 = np.linspace(0, total_1fps_frames - 1, 256).astype(int)
    frames_256 = frames_1fps[idx_256]


    # =====================================================================
    # STEP 2: Use SigLIP to retrieve top 64 frames based on text question
    # =====================================================================
    print("Step 2: Running SigLIP retrieval...")
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Convert the 256 numpy frames to PIL Images for the processor
    pil_frames = [Image.fromarray(frame) for frame in frames_256]
    
    # Process text and images
    inputs = processor(
        text=[text_query], 
        images=pil_frames, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Get the similarities between the text and the 256 images
        logits_per_text = outputs.logits_per_text.squeeze() # Shape: (256,)
        
    # Retrieve the indices of the top 64 highest scoring frames
    top_64_idx_in_256 = torch.topk(logits_per_text, 64).indices.cpu().numpy()

    # =====================================================================
    # STEP 3: Create the dictionary with mapped indices
    # =====================================================================
    print("Step 3: Generating index mapping dictionary...")
    # Map the top 64 indices (which are out of 256) back to their 
    # corresponding indices in the original 1 fps video sequence.
    selected_indices_1fps = idx_256[top_64_idx_in_256]
    
    # Sort them chronologically and remove any duplicates 
    # (duplicates could occur if the video is very short)
    selected_indices_1fps = np.unique(selected_indices_1fps)
    selected_indices_1fps.sort()
    
    mapping_dict = {}
    num_selected = len(selected_indices_1fps)
    
    for i in range(num_selected):
        current_idx = selected_indices_1fps[i]
        
        # Determine the next key to know the boundary for sampling between them
        if i < num_selected - 1:
            next_idx = selected_indices_1fps[i+1]
        else:
            # For the last selected frame, sample up to the end of the 1fps video
            next_idx = total_1fps_frames
            
        # Get all indices strictly between current_idx and next_idx
        available_between = np.arange(current_idx + 1, next_idx)
        
        # Uniformly sample up to 16 indices from the available gap
        if len(available_between) > 16:
            sampled_gap = available_between[np.linspace(0, len(available_between) - 1, 16).astype(int)]
        else:
            sampled_gap = available_between
            
        # Assign to dictionary (converting numpy array to standard Python list)
        mapping_dict[int(current_idx)] = sampled_gap.tolist()

    # Return required objects
    return frames_1fps, frames_1fps[list(mapping_dict.keys())], mapping_dict

def create_and_save_grid(frames, output_path="frame_grid.jpg"):
    # Ensure we have exactly 64 frames
    assert len(frames) == 64, "Expected exactly 64 frames for an 8x8 grid."
    
    # Check if frames are numpy arrays (from decord) and convert to tensors
    if isinstance(frames, np.ndarray) or isinstance(frames[0], np.ndarray):
        # Stack if it's a list of arrays, then convert to tensor
        frames_tensor = torch.from_numpy(np.stack(frames))
        # Numpy arrays from decord are [H, W, C]. torchvision expects [C, H, W].
        # So we permute: [B, H, W, C] -> [B, C, H, W]
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    elif isinstance(frames, torch.Tensor):
        # If it's already a tensor, make sure it's [B, C, H, W]
        if frames.shape[-1] == 3:
            frames_tensor = frames.permute(0, 3, 1, 2)
        else:
            frames_tensor = frames
    else:
        raise ValueError("Unsupported frame format. Use numpy arrays or PyTorch tensors.")

    # make_grid expects float tensors in range [0, 1] for best results, 
    # but it can handle uint8 [0, 255] if we convert it back properly. 
    # Let's convert to float and normalize to [0, 1]
    if frames_tensor.dtype == torch.uint8:
        frames_tensor = frames_tensor.float() / 255.0

    # Create the grid
    # nrow=8 means 8 images per row. Since we have 64 images, it will naturally form an 8x8 grid.
    grid_tensor = make_grid(frames_tensor, nrow=8, padding=2, pad_value=1.0) # White padding

    # Convert the grid tensor back to a PIL image and save
    grid_image = to_pil_image(grid_tensor)
    grid_image.save(output_path)
    print(f"Grid saved successfully to {output_path}")

    return grid_image

# =====================================================================
# Example Execution
# =====================================================================
if __name__ == "__main__":
    video_file = "/nfshomes/asarkar6/aditya/5qMcDQd17Y4.mp4"
    query = "What object is there on right side of person wearing white shirt?"
    
    try:
        f_1fps, f_64, frame_mapping = process_video_pipeline(video_file, query)
        
        print("\n--- Results ---")
        print(f"Total 1 fps frames extracted: {len(f_1fps)}")
        print(f"Sub-sampled frames extracted: {len(f_64)}")
        print(f"Number of keys in dictionary: {len(frame_mapping)}")
        
        # Print a sneak peek of the dictionary
        print("\nMapping Dictionary (First 3 entries):")
        for k, v in list(frame_mapping.items())[:3]:
            print(f"Selected Frame (1fps index) {k} -> Interim frames: {v}")

        # save the frid
        create_and_save_grid(f_64, "my_video_grid.jpg") 
            
    except Exception as e:
        print(f"Error: {e}")