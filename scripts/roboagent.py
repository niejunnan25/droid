import contextlib
import dataclasses
import faulthandler
import os
import select
import signal
import sys
import threading
import time
import pickle
import yaml
from datetime import datetime
from typing import Optional

# 3rd party imports
import numpy as np
import pandas as pd
import torch
import tqdm
import tyro
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip

# transformers
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer # (Note: These were imported but not used in your last script)

# Project specific imports
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from droid.robot_env import RobotEnv
from utils import crop_left_right
from robobrain.utils import prevent_keyboard_interrupt, clear_input_buffer
import timer # (Ensure this timer module is available)

faulthandler.enable()


@dataclasses.dataclass
class Args:
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    external_camera: Optional[str] = "left"
    max_timesteps: int = 1200
    max_duration: int = 300  # (Fixed typo: max_duartion -> max_duration)
    
    # --- [FIXED] Added missing Args ---
    record_dir: str = "data/episodes"
    save_deployment_data: bool = False
    # --- End Fix ---

    open_loop_horizon: int = 8
    control_frequency: int = 10
    remote_host: int = "162.105.195.74"
    remote_port: int = 8000


class RoboAgent:
    def __init__(self, args):
        
        self.args = args
        self.log_lock = threading.Lock()

        self.left_image = None
        self.right_image = None
        self.wrist_image = None
        self.joint_position = None
        self.gripper_position = None
        self.running = True
        
        # --- [FIXED] Initialized variables for thread safety at startup ---
        self.in_post_run = True  # Start in 'post_run' state to prevent threads from running early
        self.episode_start = time.time()
        self.current_instruction = None # Ensure check_subtask_success thread starts safely
        # --- End Fix ---

        self.start_time = time.time()
        self.consecutive_failed_attempts = 0

        self.env = self.init_robot()
        self.log("Created the droid env!", "info")

        self.policy = self.init_policy(self.args.remote_host, self.args.remote_port)
        self.log(f"成功连接到 {self.args.remote_host}: {self.args.remote_port}!", "info")

        self.init_vlm()
        self.log("Init RoboBrain done", "info")

        self.init_threads()
        self.log("Init threads done", "info")

        self.subtask_executed_done = False

        self.df = pd.DataFrame(columns=["success", "duration", "is_follow", "object", "language_instruction", "video_filename"])
    
    def init_robot(self, action_space: str = "joint_position", gripper_action_space: str = "position"):
        env = RobotEnv(action_space=action_space, gripper_action_space=gripper_action_space)
        return env
    
    def init_policy(self, remote_host: str, remote_port: int = 8000):
        client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)
        return client
    
    def init_vlm(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.planner = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", device_map="auto")
        self.robobrain = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", device_map="auto")
    
    def init_threads(self):
        self.camera_thread = threading.Thread(target=self.capture_images, daemon=True)
        self.camera_thread.start()

        self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_listener_thread.start()

        self.check_subtask_success_thread = threading.Thread(target=self.check_subtask_success, daemon=True)
        self.check_subtask_success_thread.start()

        self.monitor_execution_time_thread = threading.Thread(target=self.monitor_execution_time, daemon=True)
        self.monitor_execution_time_thread.start()

    def capture_images(self):
        while True:
            try:
                obs = self.env.get_observation()
                image_observations = obs["image"]

                for key in image_observations:
                    if self.args.left_camera_id in key and "left" in key:
                        self.left_image = self.process_image(image_observations[key])
                    elif self.args.right_camera_id in key and "left" in key:
                        self.right_image = self.process_image(image_observations[key])
                    elif self.args.wrist_camera_id in key and "left" in key:
                        self.wrist_image = self.process_image(image_observations[key])

                self.joint_position = np.array(obs["robot_state"]["joint_positions"])
                self.gripper_position = np.array([obs["robot_state"]["gripper_position"]])

            except Exception as e:
                self.log(f"Error in capture_images: {e}", "error")
                time.sleep(0.1)

            if not self.running:
                break

    def keyboard_listener(self):
        """Thread to listen for keyboard input"""
        while True:
            try:
                if not self.in_post_run:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        sys.stdin.readline()
                        self.log("<Enter> detected, ending this episode.", message_type="info")
                        
                        self.save_image(self.left_image, "head_image_grasp_finish")
                        self.log(f"Head camera image saved at grasp finish.", message_type="info")
                        self.save_image(self.wrist_image, "wrist_image_grasp_finish")
                        self.log(f"Wrist camera image saved at grasp finish.", message_type="info")
                        
                        self.in_post_run = True
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.log(f"Error in keyboard_listener: {e}", "error")

            if not self.running:
                break
    
    def process_image(self, img, crop_ratios=None):
        """(User-provided) Processes raw image: BGR -> RGB and optional crop."""
        if img is None:
            return None
        img = img[..., :3][..., ::-1] # Assume BGR, convert to RGB
        pil_img = Image.fromarray(img)
        if crop_ratios:
            pil_img = crop_left_right(pil_img, *crop_ratios)
        return np.array(pil_img)

    def get_current_instruction(self):
        """
        Step1: Plan which object to grasp next based on initial and current images
        (Note: Assumes self.planner.request_task is implemented elsewhere)
        """
        current_image = self.left_image
        
        # [!!!] VLM WRAPPER: 'timer' and 'self.planner.request_task' are assumed to exist [!!!]
        with timer.timer("global instruction proposal", self.log_file):
            self.current_instruction = self.planner.request_task(
                task_name="global_instruction_proposal",
                vl_inputs={
                    "images": {
                        "initial_image": self.initial_image,
                        "current_image": current_image
                    },
                    "user_prompt": self.user_prompt
                }
            )
        self.log(f"Current instruction: {self.current_instruction}", message_type="outcome")

    def check_subtask_success(self):
        """
        Step2: Check current subtask is done
        (Note: Assumes self.robobrain.request_task is implemented elsewhere)
        """
        while True:
            time.sleep(0.1) # Check at 10Hz
            try:
                if not self.in_post_run and self.current_instruction is not None:
                    
                    left_image = Image.fromarray(self.left_image) if self.left_image is not None else None
                    right_image = Image.fromarray(self.right_image) if self.right_image is not None else None
                    wrist_image = Image.fromarray(self.wrist_image) if self.wrist_image is not None else None

                    subtask_success = False

                    # [!!!] VLM WRAPPER: 'timer' and 'self.robobrain.request_task' are assumed to exist [!!!]
                    with timer.timer("check subtask success", self.log_file):
                        subtask_success = self.robobrain.request_task(
                            task_name="check subtask success",
                            vl_inputs={
                                "images": {
                                    "left_image": left_image,
                                    "wrist_image": wrist_image
                                },
                                "current_instruction": self.current_instruction
                            }
                        )
                    
                    if subtask_success:
                        self.log(f"Subtask '{self.current_instruction}' success!", message_type="outcome")
                        self.subtask_executed_done = True
                        self.consecutive_failed_attempts = 0
                        self.in_post_run = True  # Mark subtask as done
                        self.current_instruction = None # Clear current task
                    
                    elif self.consecutive_failed_attempts >= 3:
                        self.log("Three consecutive failed attempts. Resetting robot.", message_type="outcome")
                        self.in_post_run = True # Mark subtask as failed
                        self.consecutive_failed_attempts = 0
                        self.current_instruction = None # Clear current task
            
            except Exception as e:
                self.log(f"Error in check_subtask_success: {e}", "error")

            if not self.running:
                break

    def check_complete(self) -> bool:
        """
        Step3 : Check task is completed
        (Note: Assumes self.planner.request_task is implemented elsewhere)
        """
        current_image = self.left_image
        
        # [!!!] VLM WRAPPER: 'timer' and 'self.planner.request_task' are assumed to exist [!!!]
        with timer.timer("prompt completion check", self.log_file):
            task_completed = self.planner.request_task(
                task_name="prompt_completion_check",
                vl_inputs={
                    "images": {
                        "initial_mage": self.initial_image, # (Note: typo 'initial_mage' from your original code)
                        "current_image": current_image
                    },
                    "user_prompt": self.user_prompt
                }
            )
            
        if task_completed:
            self.log(f"User prompt '{self.user_prompt}' is completed.", message_type="outcome")
        else:
            self.log(f"User prompt '{self.user_prompt}' is not completed, continuing to grasp.", message_type="outcome")
            
        return task_completed
    
    def monitor_execution_time(self):
        """Thread to monitor task execution time"""
        while True:
            try:
                # Use a longer sleep time, e.g., 1 second
                time.sleep(1.0) 
                
                if not self.in_post_run:
                    elapsed_time = time.time() - self.episode_start
                    if elapsed_time > self.args.max_duration: # (Fixed typo: max_duartion -> max_duration)
                        self.log(f"Episode exceeded maximum duration of {self.args.max_duration}s", "warning")
                        self.in_post_run = True

            except Exception as e:
                 self.log(f"Error in monitor_execution_time: {e}", "error")
            
            if not self.running:
                break

    def init_episode(self, manual=False):
        """(Note: Assumes self.planner.set_logging is implemented elsewhere)"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_episode_dir = os.path.join(self.args.record_dir, timestamp) # (Fixed: uses self.args.record_dir)
            os.makedirs(self.current_episode_dir, exist_ok=True)

            self.log_file_path = os.path.join(self.current_episode_dir, 'run.log')
            self.log_file = open(self.log_file_path, 'w')

            if manual:
                self.log(f"Episode starts, using manual mode.", message_type="info")
            else:
                self.log(f"Episode starts, using planner mode.", message_type="info")

            self.img_dir = os.path.join(self.current_episode_dir, 'images')
            os.makedirs(self.img_dir, exist_ok=True)

            if not manual:
                # [!!!] VLM WRAPPER: 'self.planner.set_logging' is assumed to exist [!!!]
                self.planner.set_logging(self.log_file, self.img_dir)

            if self.args.save_deployment_data: # (Fixed: uses self.args.save_deployment_data)
                self.right_cam_buffer, self.rgbm_buffer, self.state_buffer_save, self.action_buffer = [], [], [], []

                width, height = 640, 480
                fps = 20
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(self.current_episode_dir, 'video.mp4')
                self.video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc, fps, (width * 3, height))
        except Exception as e:
            self.log(f"Error in init_episode: {e}", "error")
            # This is critical, if episode fails to init, we should probably stop
            self.running = False 

    def reset_flags(self):
        self.log("Resetting episode flags.", "debug")
        self.update_target = True # (Note: This flag is set but never used in your logic)
        self.in_post_run = False # <--- This is the crucial flag that starts the threads' work
        self.episode_start = time.time()
        self.consecutive_failed_attempts = 0
        self.subtask_executed_done = False # <--- Also crucial

    def run_manual(self):
        pass

    def run_agent(self):
        self.user_prompt = input("""Please enter your instruction\n>>>  """)

        self.initial_image = self.left_image
        if self.initial_image is None:
            self.log("Failed to get initial_image from camera. Exiting.", "error")
            return

        self.log(f"User prompt: {self.user_prompt}", message_type="outcome")

        while True: # Outer loop (for entire task)
            self.time_step = 0
            self.init_episode(manual=False)

            if self.check_complete():
                self.close_episode()
                break # Task was already complete

            while True: # Inner loop (for sub-tasks)
                
                # 步骤1：规划
                subtask = self.get_current_instruction()
                
                # 步骤2：重置
                self.reset_flags()
                clear_input_buffer() # (Note: Assumes clear_input_buffer is defined)

                # 步骤3：执行
                self.execute(subtask)
                
                # 步骤4：检查总任务
                task_completed = self.check_complete()
                if task_completed:
                    break # Exit inner sub-task loop
            
            self.close_episode()
            self.log("Task finished, exiting run_agent.", "info")
            break # Exit outer task loop (as requested)
    
    def execute(self, subtask: str):
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        
        # --- [FIXED] Added timestep counter for max_timesteps check ---
        timestep = 0 
        self.log(f"Executing subtask: {subtask}", "info")

        while not self.subtask_executed_done and not self.in_post_run:
            
            # --- [FIXED] Check for max_timesteps ---
            if timestep >= self.args.max_timesteps:
                self.log(f"Subtask execution reached max_timesteps ({self.args.max_timesteps}).", "warning")
                self.in_post_run = True # Force stop
                break
            # --- End Fix ---

            start_time = time.time()
            try:
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    request_data = {
                        "observation/image": image_tools.resize_with_pad(self.left_image, 224, 224),
                        "observation/wrist_image": image_tools.resize_with_pad(self.wrist_image, 224, 224),
                        "observation/state": np.concatenate((self.joint_position, self.gripper_position), axis=0),
                        "prompt": subtask,
                    }

                    with prevent_keyboard_interrupt(): # (Note: Assumes this util exists)
                        s_time = time.time()
                        pred_action_chunk = self.policy.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]
                        self.log(f"Policy inference time: {time.time()- s_time:.4f} s", "debug")

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Gripper action thresholding
                if action[-1].item() > 0.20:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                self.env.step(action)
                timestep += 1 # --- [FIXED] Increment timestep counter ---

                elapsed_time = time.time() - start_time
                sleep_time = (1 / self.args.control_frequency) - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.log("KeyboardInterrupt in execute loop. Stopping.", "info")
                self.in_post_run = True # Gracefully stop
                break
            except Exception as e:
                self.log(f"\n[FATAL] Error in execute loop: {e}", "error")
                self.in_post_run = True # Force stop on error
                break

    def log(self, message: str, message_type: str = "info"):
        """ (User-provided) A thread-safe logger. """
        with self.log_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_message = f"[{timestamp}] [{message_type.upper()}] {message}"
            
            print(log_message)
            
            if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
                try:
                    self.log_file.write(log_message + "\n")
                    self.log_file.flush()
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [LOG_ERROR] Failed to write to log file: {e}")

    # --- [FIXED] Added missing helper functions ---
    
    def save_image(self, image, filename_prefix):
        """Saves an image to the current episode's image directory."""
        if image is None:
            self.log(f"Cannot save image {filename_prefix}, image is None.", "warning")
            return
            
        if not hasattr(self, 'img_dir') or not self.img_dir:
             self.log(f"Cannot save image, img_dir is not set (episode not initialized?).", "error")
             return
             
        try:
            img_path = os.path.join(self.img_dir, f"{filename_prefix}.png")
            Image.fromarray(image).save(img_path)
            self.log(f"Image saved to {img_path}", "debug")
        except Exception as e:
            self.log(f"Failed to save image {filename_prefix}: {e}", "error")

    def close_episode(self):
        """Cleans up resources at the end of an episode."""
        self.log("Closing episode.", "info")
        
        # Release video writer
        if hasattr(self, 'video_writer') and self.video_writer:
             self.video_writer.release()
             self.log("Video writer released.", "debug")
             self.video_writer = None

        # Close log file
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()
            self.log("Log file closed.", "debug")
            self.log_file = None
    # --- End Fix ---

if __name__ == '__main__':
    # This is how you would run your code
    # 1. Parse arguments
    args = tyro.cli(Args)
    
    # 2. Create the agent (this will initialize threads, env, etc.)
    agent = RoboAgent(args)
    
    # 3. Run the agent
    try:
        agent.run_agent()
    except KeyboardInterrupt:
        agent.log("Shutdown requested by user.", "info")
    except Exception as e:
        agent.log(f"An uncaught exception occurred: {e}", "error")
    finally:
        # Ensure threads are signalled to stop
        agent.running = False
        agent.close_episode() # Final cleanup
        agent.log("RoboAgent shutdown complete.", "info")
        # Exit to ensure daemon threads are killed
        sys.exit(0)