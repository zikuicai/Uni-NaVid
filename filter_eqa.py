import json
import os

def filter_open_eqa():
    # --- Configuration ---
    input_file = 'open-eqa.json'
    output_file = 'open_eqa_q.json'
    
    # Target directory path
    target_dir_path = '/fs/nexus-projects/tiamat-benchmark/tiamat_ws/sequential_eqa/hm3d/hm3d_scenes'
    
    # --- Step 1: Get Valid Scene IDs ---
    print(f"Scanning directories in {target_dir_path}...")
    
    try:
        # Get all directory names in the target path
        all_dirs = os.listdir(target_dir_path)
        
        # Create a set of the last 11 characters of each directory for O(1) lookup
        # We filter to ensure we are only looking at directories, not files
        valid_scenes = set()
        for d in all_dirs:
            full_path = os.path.join(target_dir_path, d)
            if os.path.isdir(full_path):
                # Store only the last 11 characters
                valid_scenes.add(d[-11:])
                
    except FileNotFoundError:
        print(f"Error: The directory {target_dir_path} was not found.")
        return

    # --- Step 2: Process the JSON ---
    print(f"Processing {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    filtered_data = []

    # Iterate through each item in the input list
    for item in data:
        # specific error handling in case a key is missing in a row
        try:
            ep_history = item.get('episode_history', '')
            
            # Extract last 11 characters of the episode history
            # Ensure the string is at least 11 chars long
            if len(ep_history) >= 11:
                scene_candidate = ep_history[-11:]
                
                # Check for match
                if scene_candidate in valid_scenes:
                    # Create the dictionary entry
                    entry = {
                        "question": item.get('question'),
                        "answer": item.get('answer'),
                        "scene": scene_candidate
                    }
                    filtered_data.append(entry)
                    
        except AttributeError:
            # Handle cases where data might not be a clean dictionary
            continue

    # --- Step 3: Output Results ---
    count = len(filtered_data)
    print(f"Total matching entries found: {count}")
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    filter_open_eqa()