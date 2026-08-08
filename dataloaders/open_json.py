import ijson
import pdb

json_path = "/work/YamadaU/asarkar/agent2vec_outputs/thinking_texts_kinetics.json"

count = 0
with open(json_path, "rb") as f:
    # 'item' tells ijson to yield each individual dict from the root list
    for record in ijson.items(f, "item", multiple_values=True):
        label = record["label"]
        video_path = record["video_path"]
        conversations = record["conversations"]
        
        # Process one dictionary at a time...
        count+=1

print(count)