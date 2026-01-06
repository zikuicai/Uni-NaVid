import os
import re
import numpy as np
import torch
import cv2
import pandas as pd
import habitat_sim
import time
import glob
from scipy.spatial.transform import Rotation as R


# UniNaVid Imports
from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# --- CONFIGURATION ---
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2" 


# Set Seeds
seed = 30
torch.manual_seed(seed)
np.random.seed(seed)

# --- VISUALIZATION UTILS ---
def draw_traj_arrows_fpv(
    img,
    actions,
    arrow_len=60,       
    arrow_gap=15,       
    arrow_color=(0, 255, 0), # Green for immediate action
    future_color=(255, 255, 0), # Cyan for queued actions
    arrow_thickness=4,  
    tipLength=0.35,
    stop_color=(0, 0, 255),  
    stop_radius=15      
):
    """
    Draws navigation arrows on the image (BGR format).
    Args:
        actions: List of strings (e.g. ["forward", "left", "forward"])
    """
    out = img.copy()
    h, w = out.shape[:2]
    
    # Start drawing from bottom center
    base_x, base_y = w // 2, int(h * 0.90)

    for i, action in enumerate(actions):
        waypoint = None
        
        # Color Logic: First action is Green, queued actions are Cyan
        current_color = arrow_color if i == 0 else future_color

        if action == "stop":
            waypoint = [0.0, 0.0, 0.0]
        elif action in ["forward", "move_forward"]:
            waypoint = [0.5, 0.0, 0.0]
        elif action in ["left", "turn_left"]:
            waypoint = [0.0, 0.0, -np.deg2rad(45)] # Exaggerated angle for visual clarity
        elif action in ["right", "turn_right"]:
            waypoint = [0.0, 0.0, np.deg2rad(45)]
        else:
            continue  

        x, y, yaw = waypoint
        
        # Calculate start point for this arrow (stacking upwards if multiple)
        # i=0 is at bottom, i=1 is above it, etc.
        start_y = int(base_y - i * (arrow_len + arrow_gap))
        
        # Stop drawing if we go off screen
        if start_y < 0: break

        start = (int(base_x), start_y)

        if action == "stop":
            cv2.circle(out, start, stop_radius, stop_color, -1) 
            cv2.putText(out, "STOP", (start[0] + 20, start[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stop_color, 2)
        else:
            # Calculate end point based on yaw
            end = (
                int(start[0] + arrow_len * np.sin(yaw)),
                int(start[1] - arrow_len * np.cos(yaw))
            )
            cv2.arrowedLine(out, start, end, current_color, arrow_thickness, tipLength=tipLength)
            
            # Optional: Add text label for clarity
            # label = action[0].upper()
            # cv2.putText(out, label, (end[0] + 10, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
    
    return out

class UniNaVid_Agent:
    def __init__(self, model_path):
        print("Initialize UniNaVid Agent...")
        self.model_path = model_path
        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path)
        )

        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            "of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. Analyze this series of images to determine your next four actions."
            "The predicted action should be one of the following: forward, left, right, or stop."
            "If you feel the video and image provide enough information to answer the task, respond with 'stop'. You do not have to reach a specific location."
        )
        
        self.rgb_list = []
        self.pending_action_list = []
        self.current_scene_id = None
        self.reset(force_wipe=True)
        print("Initialization Complete")

    def reset(self, scene_id=None, force_wipe=False):
        self.pending_action_list = []
        if force_wipe or (scene_id is not None and scene_id != self.current_scene_id):
            if scene_id:
                print(f"New Scene Detected ({scene_id}). Wiping Memory.")
            self.current_scene_id = scene_id
            self.model.get_model().initialize_online_inference_nav_feat_cache()
            self.model.get_model().new_frames = 0
            self.rgb_list = []

    def process_images(self, rgb_list):
        if len(rgb_list) == 0: return None
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        return [video]

    def predict_inference(self, prompt, mode="navigation"):
        VIDEO_START = "<video_special>"
        VIDEO_END = "</video_special>"
        IMAGE_START = "<image_special>"
        IMAGE_END = "</image_special>"
        NAVIGATION = "[Navigation]"
        IMG_SEP = "<image_sep>"
        
        def get_tok(t): return self.tokenizer(t, return_tensors="pt").input_ids[0][1:].cuda()
        tokens = { 'vid_s': get_tok(VIDEO_START), 'vid_e': get_tok(VIDEO_END),
            'img_s': get_tok(IMAGE_START), 'img_e': get_tok(IMAGE_END),
            'nav': get_tok(NAVIGATION), 'sep': get_tok(IMG_SEP) }

        has_new_images = len(self.rgb_list) > 0
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt.replace('<image>', '').replace(DEFAULT_IMAGE_TOKEN, '') if has_new_images else prompt.replace('<image>', '').replace(DEFAULT_IMAGE_TOKEN, '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        if has_new_images:
            token_prompt = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
            indices_to_replace = torch.where(token_prompt == -200)[0]
            new_list = []
            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                new_list.append(token_prompt[:idx])
                new_list.append(tokens['vid_s'])
                new_list.append(tokens['sep'])
                new_list.append(token_prompt[idx:idx + 1])
                new_list.append(tokens['vid_e'])
                new_list.append(tokens['img_s'])
                new_list.append(tokens['img_e'])
                if mode == "navigation": new_list.append(tokens['nav'])
                token_prompt = token_prompt[idx + 1:]
                indices_to_replace = torch.where(token_prompt == -200)[0]
            if token_prompt.numel() > 0: new_list.append(token_prompt)
            input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        else:
            input_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids.cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = [] 

        with torch.inference_mode():
            self.model.update_prompt([[prompt.replace(DEFAULT_IMAGE_TOKEN, '')]])
            output_ids = self.model.generate(
                input_ids, images=imgs, do_sample=True, temperature=0.2,
                max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        return outputs.strip().replace(stop_str, "").strip()

    def act(self, observations):
        # scene_id = observations.get('scene_id')
        # self.reset(scene_id=scene_id)
        self.rgb_list.append(observations['rgb'])

        if len(self.pending_action_list) > 0:
            # print(f"Executing Pending Action (Queue: {len(self.pending_action_list)})")
            return {"action": self.pending_action_list.pop(0)}

        instruction = observations['instruction']
        prompt = self.promt_template.format(instruction)
        outputs = self.predict_inference(prompt, mode="navigation")
        
        actions = outputs.split(" ")
        valid_actions = ["forward", "left", "right", "stop"]
        clean_actions = [a for a in actions if a in valid_actions]
        if not clean_actions: clean_actions = ["stop"]

        if "stop" in clean_actions or clean_actions[0] == "stop":
            print("Agent stopped. Generating Answer...")
            ans_prompt = f"Your task is: '{instruction}'. Answer the question based on the video history. Answer according to the format specified in the question."
            self.rgb_list.append(observations['rgb']) 
            answer = self.predict_inference(ans_prompt, mode="answering")
            return { "action": "stop", "text_answer": answer, "finished": True }

        self.pending_action_list.extend(clean_actions)
        return {"action": self.pending_action_list.pop(0)}

# --- HABITAT SETUP ---
def configure_habitat(scene_path):
    sim_settings = { "width": 640, "height": 480, "scene": scene_path, "default_agent": 0, "sensor_height": 1.25, "color_sensor": True }
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = "/fs/nexus-projects/tiamat-benchmark/tiamat_ws/sequential_eqa/procthor/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
    cfg.scene_id = scene_path
    cfg.enable_physics = False
    sensor_specs = []
    color_sensor = habitat_sim.CameraSensorSpec()
    color_sensor.uuid = "color_sensor"
    color_sensor.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
    color_sensor.position = [0.0, sim_settings["sensor_height"], 0.0]
    sensor_specs.append(color_sensor)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }
    return habitat_sim.Configuration(cfg, [agent_cfg])


# --- ASSUMING IMPORTS FOR AGENT AND UTILS ARE AVAILABLE ---
# from agent_module import UniNaVid_Agent
# from habitat_utils import configure_habitat
# from visualization import draw_traj_arrows_fpv
# MODEL_PATH = "path/to/model.pt"

def get_rotation_string(quat):
    """Helper to format quaternion or rotation for logging."""
    # quat is usually [x, y, z, w] in habitat or [w, x, y, z] depending on version.
    # We will simply log the raw values for precision.
    return f"[{quat.x:.4f}, {quat.y:.4f}, {quat.z:.4f}, {quat.w:.4f}]"

if __name__ == '__main__':
    # --- Configuration ---
    BASE_DIR = "/fs/nexus-projects/tiamat-benchmark/tiamat_ws"
    QUESTION_DIR = os.path.join(BASE_DIR, "tiamat-benchmark/data/procthor")
    SCENE_ROOT = os.path.join(BASE_DIR, "sequential_eqa/procthor/ai2thor-hab/configs/scenes/ProcTHOR/1")
    INIT_POS_CSV = os.path.join(BASE_DIR, "sequential_eqa/procthor/procthor_qa_verified/init_poses/init_pose_procthor.csv")
    
    # Output Directories
    TRACE_ROOT = "procthor/traces"
    RESULTS_DIR = "procthor/results"
    
    if not os.path.exists(TRACE_ROOT): os.makedirs(TRACE_ROOT)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # --- Load Initialization Data ---
    start_pos_df = pd.read_csv(INIT_POS_CSV)
    
    # Global Counters
    total_correct = 0
    total_questions = 0

    # --- Iterate through all Question CSVs ---
    csv_files = glob.glob(os.path.join(QUESTION_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")

    # Dictionary to cache simulator instances if needed (optional optimization), 
    # but simplest is to re-init or re-configure if scene changes.
    sim = None
    current_scene_id = None
    agent = UniNaVid_Agent(MODEL_PATH)
    hab_map = {"forward": "move_forward", "left": "turn_left", "right": "turn_right", "stop": "stop"}

    for csv_path in csv_files:
        print(f"\nProcessing file: {os.path.basename(csv_path)}")
        q_df = pd.read_csv(csv_path)
        
        # Prepare list to store results for this CSV
        results_data = []

        for index, row in q_df.iterrows():
            question = row['question']
            correct_ans = str(row['answer'])
            scene_id = str(os.path.basename(csv_path))[:-4]  # Assuming scene_id is derived from filename
            
            # --- 1. Setup Simulator for Scene ---
            if scene_id != current_scene_id:
                if sim is not None:
                    sim.close()
                
                scene_path = os.path.join(SCENE_ROOT, f"ProcTHOR-{scene_id[9:]}.scene_instance.json")
                if not os.path.exists(scene_path):
                    print(f"Error: Scene file not found at {scene_path}")
                    continue
                    
                cfg = configure_habitat(scene_path)
                sim = habitat_sim.Simulator(cfg)
                current_scene_id = scene_id
                print(f"  > Loaded Scene: {scene_id}")

            # --- 2. Initialize Agent Position ---
            # Find start pos for this scene
            start_row = start_pos_df[start_pos_df['Scene'] == scene_id]
            if start_row.empty:
                print(f"  > Warning: No start position found for scene {scene_id}. Skipping.")
                continue
            
            start_vals = start_row.iloc[0]
            start_pos = [start_vals['init_x'], start_vals['init_y'], start_vals['init_z']]

            agent_sim = sim.get_agent(0)
            state = agent_sim.get_state()
            state.position = np.array(start_pos)
            # You might need to set rotation here if start_pos_df has it, 
            # otherwise it defaults to 0 or whatever previous state was.
            # state.rotation = ... 
            agent_sim.set_state(state)
            agent.reset(scene_id=scene_id)

            # --- 3. Run Episode ---
            print(f"    [Q{index}] {question}")
            
            # Trace setup
            trace_dir = os.path.join(TRACE_ROOT, scene_id, str(index))
            if not os.path.exists(trace_dir): os.makedirs(trace_dir)
            trace_file_path = os.path.join(trace_dir, "trace.txt")
            trace_lines = []

            # Metrics setup
            start_time = time.time()
            dist_traveled = 0.0
            prev_position = np.array(start_pos)
            
            done = False
            steps = 0
            max_steps = 2000
            pred_ans = ""
            
            # Record Initial State
            init_rot_str = get_rotation_string(state.rotation)
            trace_lines.append(f"START | Pos: {start_pos} | Rot: {init_rot_str}\n")

            while not done and steps < max_steps:
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:, :, :3]
                
                # Agent Act
                result = agent.act({
                    'instruction': question,
                    'rgb': rgb,
                    'scene_id': scene_id
                })
                action_name = result['action']
                
                # Execute Action in Sim
                if action_name in hab_map and action_name != "stop":
                    sim.step(hab_map[action_name])
                elif action_name == "stop":
                    done = True

                # Get New State
                state = agent_sim.get_state()
                curr_position = state.position
                curr_rot_str = get_rotation_string(state.rotation)

                # Calculate Distance
                step_dist = np.linalg.norm(curr_position - prev_position)
                dist_traveled += step_dist
                prev_position = curr_position

                # Log Trace: "Action | New XYZ | New Rotation"
                # Using a pipe delimiter for readability, formatted as requested
                log_line = f"{action_name}, Pos: [{curr_position[0]:.4f}, {curr_position[1]:.4f}, {curr_position[2]:.4f}], Rot: {curr_rot_str}\n"
                trace_lines.append(log_line)

                # Check Finish
                if result.get('finished'):
                    pred_ans = result.get('text_answer', "")
                    done = True
                
                steps += 1

            # End of Episode Processing
            elapsed_time = time.time() - start_time
            
            # Write Trace File
            with open(trace_file_path, "w") as f:
                f.writelines(trace_lines)

            # Check Accuracy
            is_correct = (str(pred_ans).strip().lower() == correct_ans.strip().lower())
            if is_correct:
                total_correct += 1
            total_questions += 1

            # Append to Results
            results_data.append({
                "scene_file_name": scene_id,
                "question": question,
                "answer": correct_ans,
                "answer_pred": pred_ans,
                "time": elapsed_time,
                "distance": dist_traveled
            })

        # --- 4. Save CSV Output for this file ---
        output_csv_name = f"results_{os.path.basename(csv_path)}"
        output_csv_path = os.path.join(RESULTS_DIR, output_csv_name)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_csv_path, index=False)
        print(f"  > Saved results to: {output_csv_path}")

    # --- Final Cleanup and Report ---
    if sim:
        sim.close()

    accuracy = (total_correct / total_questions) if total_questions > 0 else 0
    print("-" * 50)
    print("FINAL RESULTS")
    print("-" * 50)
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct:   {total_correct}")
    print(f"Accuracy:        {accuracy:.2%} ({total_correct}/{total_questions})")
    print("-" * 50)