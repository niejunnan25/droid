import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional
import cv2
from utils import crop_left_right
faulthandler.enable()
import pickle
# Python standard library imports
import argparse
import os
import select
import sys
import threading
import time
from datetime import datetime
import yaml

import timer

import threading

import torch
from PIL import Image
import time

from robobrain.utils import prevent_keyboard_interrupt, clear_input_buffer


from utils import parse_llm_output_to_list_regex, extract_boolean_answer, process_image


@dataclasses.dataclass
class Args:
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    external_camera: Optional[str] = "left"
    max_timesteps: int = 1200
    max_duration : int = 300
    open_loop_horizon: int = 8
    control_frequency : int = 10
    remote_host : str = "162.105.195.74"
    remote_port: int = 8000

    record_dir : str = "record"
    save_deployment_data : bool = False

    action_space : str = "joint_position"
    gripper_action_space : str = "position"

    max_subtask_timesteps : int = 400

class RoboAgent:
    def __init__(self, args):
        
        self.args = args
        self.log_lock = threading.Lock()

        self.left_image = None
        self.right_image = None
        self.wrist_image = None

        self.image_lock = threading.Lock()
        self.current_subtask_lock = threading.Lock()
        self.subtask_executed_done_lock = threading.Lock()
        self.post_run_lock = threading.Lock()

        self.current_subtask = None   # <--- 在这里添加这一行

        self.joint_position = None
        self.gripper_position = None
        self.running = True

        self.in_post_run = True  # 设为True，防止线程在 agent 准备好前错误地运行
        self.episode_start = time.time()

        self.start_time = time.time()
        self.consecutive_failed_attempts = 0

        self.env = RobotEnv(action_space=self.args.action_space, gripper_action_space=self.args.gripper_action_space)
        self.log("Created the droid env!", message_type="info")

        self.policy = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)
        self.log(f"Successfully connected to {self.args.remote_host}: {self.args.remote_port}!", "info")

        self.init_vlm()
        self.log("Init RoboBrain done", "info")

        self.init_threads()
        self.log("Init threads done", "info")

        self.subtask_executed_done = False

        self.df = pd.DataFrame(columns=["success", "duration", "is_follow", "object", "language_instruction", "video_filename"])

        self.vlm = None
    
    def init_vlm(self):
        from scripts.robobrain import RoboBrain
        self.vlm = RoboBrain(8001)
    

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

                    # 共享变量的写入要用锁保护
                    with self.image_lock:
                        if self.args.left_camera_id in key and "left" in key:
                            self.left_image = process_image(image_observations[key])
                        elif self.args.right_camera_id in key and "left" in key:
                            self.right_image = process_image(image_observations[key])
                        elif self.args.wrist_camera_id in key and "left" in key:
                            self.wrist_image = process_image(image_observations[key])

                        # 状态信息统一被 image_lock保护
                        self.joint_position = np.array(obs["robot_state"]["joint_positions"])
                        self.gripper_position = np.array([obs["robot_state"]["gripper_position"]])

            except Exception as e:
                self.log(f"Error in capture_images: {e}", "error")
                time.sleep(0.1)

            if self.running == False:
                break            


    def keyboard_listener(self):
        """Thread to listen for keyboard input"""
        while True:
            try:
                with self.post_run_lock:
                    if self.in_post_run == False:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            sys.stdin.readline()
                            self.log("<Enter> detected, ending this episode.", message_type="info")
                            
                            self.in_post_run = True
                    else:
                        time.sleep(0.1)
            except Exception as e:
                self.log(f"Error in keyboard_listener: {e}", "error")

            if self.running == False:
                break
    
    def monitor_execution_time(self):
        """Thread to monitor task execution time"""
        while True:
            try:
                # Use a longer sleep time, e.g., 1 second
                time.sleep(1.0) 
                with self.post_run_lock:
                    if self.in_post_run == False:
                        elapsed_time = time.time() - self.episode_start
                        if elapsed_time > self.args.max_duration:
                            self.log(f"Episode exceeded maximum duration of {self.args.max_duration}s", "warning")
                            self.in_post_run = True

            except Exception as e:
                 self.log(f"Error in monitor_execution_time: {e}", "error")
            
            # TODO: 这是在干嘛?
            if self.running == False:
                break

    # 线程函数， 注意对共享变量的保护
    def check_subtask_success(self):

        while True:
            current_subtask_local = None

            with self.post_run_lock:
                is_post_run = self.in_post_run

            try:
                if is_post_run == False:
                    
                    with self.current_subtask_lock:
                        current_subtask_local = self.current_subtask

                    if current_subtask_local is not None:
                        
                        with self.image_lock:
                            current_image = self.left_image
                        
                        if current_image is None:
                            self.log("check_subtask_success: 图像尚未准备好, 跳过此次检查.", "warning")
                            time.sleep(1)
                            continue # 跳过此轮循环，稍后重试

                        # 7. 执行 VLM 检查 (使用局部变量)
                        with timer.timer("subtask complete check", self.log_file):
                            subtask_completed_str : str = self.vlm.request_task(
                                task_name="subtask_complete_check",
                                vl_inputs={
                                    "images": {
                                        "current_image": current_image
                                    },
                                    "user_prompt": current_subtask_local
                                }
                            )

                        with self.subtask_executed_done_lock:
                            self.subtask_executed_done : bool | None = extract_boolean_answer(subtask_completed_str)
                            subtask_executed_done = self.subtask_executed_done
                        
                        if subtask_executed_done == None:
                            self.log("Error, subtask checking did not output in the expected format", "error")
                        
                        elif subtask_executed_done:
                            self.log(f"Subtask '{current_subtask_local}' is completed.", message_type="outcome")
                        
                        else:
                            self.log(f"Subtask '{current_subtask_local}' is not completed, continue!", message_type="outcome")
            
            except Exception as e:
                self.log(f"Error in check_subtask_success: {e}", "error")

            time.sleep(3)

            if self.running == False:
                break

            # # TODO:这是不是要由主线程去控制呢?
            # if self.consecutive_failed_attempts >= 3:
            #     self.log("Three consecutive failed attempts. Resetting robot.", message_type="outcome")

    # 主函数
    # V2 AGENT: This function REPLACES get_global_instruction
    def get_next_dynamic_step(self):
        """
        Calls the VLM to decide the next step based on *memory* and current vision.
        """
        with self.image_lock:
            current_image = self.left_image
        
        if current_image is None:
            self.log("get_next_dynamic_step: No image available.", "error")
            return None 

        with timer.timer("dynamic step proposal", self.log_file):
            # 注意：我们传入了 self.execution_history (记忆)
            next_step_str : str = self.vlm.request_task(
                task_name="dynamic_next_step_proposal",
                vl_inputs={
                    "images": { "current_image": current_image },
                    "user_prompt": self.user_prompt, # 从 self 获取全局指令
                    "execution_history": self.execution_history # V2: 传入记忆
                }
            )
        
        # 清理 VLM 的输出
        next_step_str = next_step_str.strip()
        if next_step_str.startswith("- "):
            next_step_str = next_step_str[2:]
        
        if "DONE" in next_step_str.upper():
            self.log("Dynamic Agent reports: DONE", "outcome")
            return "DONE"
        
        self.log(f"Dynamic Agent decided next step: {next_step_str}", "info")
        return next_step_str

    def check_global_complete(self) -> bool:
        
        with self.image_lock:
            current_image = self.left_image

        # self.initial_image 不是共享变量, 对他的读取不需要锁保护
        
        with timer.timer("global task complete check", self.log_file):
            # TODO: 实现这个方法, 在请求任务的时候就返回 bool
            task_completed_str : bool = self.vlm.request_task(
                task_name="global_task_complete_check",
                vl_inputs={
                    "images": {
                        "initial_image": self.initial_image,
                        "current_image": current_image
                    },
                    "user_prompt": self.user_prompt
                }
            )
        
        task_completed : bool | None = extract_boolean_answer(task_completed_str)

        if task_completed == None:
            raise "Error, global checking did not output in the expected format."
            
        if task_completed:
            self.log(f"User prompt '{self.user_prompt}' is completed.", message_type="outcome")
        else:
            self.log(f"User prompt '{self.user_prompt}' is not completed, continuing to grasp.", message_type="outcome")
            
        return task_completed
    
    ################################################ 线程函数 (Thread Functions) #################################################






    ################################################ 执行函数 (Execution Functions) ###############################################
    def init_episode(self):
        # 这个是 init_episode, 在执行之前就率先调用
        """(Note: Assumes self.planner.set_logging is implemented elsewhere)"""
        try:
            # 以 record_dir + timestamp 创建一个文件夹
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_episode_dir = os.path.join(self.args.record_dir, timestamp)
            os.makedirs(self.current_episode_dir, exist_ok=True)

            # 当前的 log_name "record_dir + timestamp + run.log"
            self.log_file_path = os.path.join(self.current_episode_dir, 'run.log')
            self.log_file = open(self.log_file_path, 'w')
            
            # 在以 record_dir + timestamp 下创建一个 images 文件夹
            self.img_dir = os.path.join(self.current_episode_dir, 'images')
            os.makedirs(self.img_dir, exist_ok=True)

            """
            result:
            |--timestamp_run.log
            |--images

            """

            # TODO: 重写这个 set_logging 函数
            self.planner.set_logging(self.log_file, self.img_dir)

            # # TODO: 怎么保存视频？
            # if self.args.save_deployment_data: 
            #     self.right_cam_buffer, self.rgbm_buffer, self.state_buffer_save, self.action_buffer = [], [], [], []

            #     width, height = 640, 480
            #     fps = 20
            #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #     video_path = os.path.join(self.current_episode_dir, 'video.mp4')
            #     self.video_writer = cv2.VideoWriter(
            #         video_path,
            #         fourcc, fps, (width * 3, height))
                
        except Exception as e:
            self.log(f"Error in init_episode: {e}", "error")
            # This is critical, if episode fails to init, we should probably stop
            self.running = False 

    def reset_flags(self):
        self.log("Resetting episode flags.", "debug")
        with self.post_run_lock:
            self.in_post_run = False # <--- This is the crucial flag that starts the threads' work
        self.episode_start = time.time()
        self.consecutive_failed_attempts = 0
        self.subtask_executed_done = False # <--- Also crucial

    def run_agent(self):

        self.user_prompt = input("""Please enter your instruction\n>>>  """)

        with self.image_lock:
            self.initial_image = self.left_image.copy()
        if self.initial_image is None:
            self.log("Failed to get initial_image from camera. Exiting.", "error")
            return
        
        self.log(f"User prompt: {self.user_prompt}", "info")
        self.init_episode() # 初始化日志

        # V2 AGENT: 重置记忆
        self.execution_history = [] 

        # ------------------- V2 AGENT LOOP -------------------
        while True: # V2: This is now the *only* loop
            
            # 1. 检查全局任务是否完成
            # 仍然需要这个函数作为最终的“退出”检查
            if self.check_global_complete():
                self.log("Global completion check is TRUE. Task finished.", "outcome")
                break # 退出主循环

            # 2. VLM 动态规划，决定下一步 (基于记忆)
            try:
                next_subtask = self.get_next_dynamic_step()
            except Exception as e:
                self.log(f"Error in VLM dynamic proposal: {e}", "error")
                break # VLM 失败，退出

            if next_subtask is None or "DONE" in next_subtask:
                self.log("VLM planner returned DONE. Exiting.", "info")
                break # VLM 认为任务已完成

            # 3. 设置并执行子任务
            with self.current_subtask_lock:
                self.current_subtask = next_subtask
            
            self.reset_flags() # 重置 subtask_executed_done = False
            clear_input_buffer()

            self.execute(next_subtask) # VLA (快小脑) 开始执行

            # 4. 记录结果 (更新记忆)
            with self.subtask_executed_done_lock:
                success = self.subtask_executed_done
            
            # 检查是否是人为中断
            with self.post_run_lock:
                in_post_run = self.in_post_run
            
            if in_post_run and not success:
                try:
                    # 检查是否有键盘输入，区分“超时”和“用户按键”
                    if select.select([sys.stdin], [], [], 0)[0]:
                        self.log("User interrupt detected after execute. Stopping task.", "info")
                        break # 用户按键，彻底终止
                except:
                    pass # 忽略 select 错误

            # 现在，更新记忆
            if success:
                self.log(f"Outcome for '{next_subtask}': Success", "info")
                self.execution_history.append(f"{next_subtask} -> Success")
            else:
                # 如果 execute 结束了，但 success 不是 True，
                # 这就意味着是超时 (max_subtask_timesteps) 导致的失败
                self.log(f"Outcome for '{next_subtask}': Failure (Timeout)", "warning")
                self.execution_history.append(f"{next_subtask} -> Failure")
            
            # 循环回到第 1 步，VLM 会看到这个新的"Failure"或"Success"
        # -----------------------------------------------------
        
        self.close_episode()
        self.log("Task finished, exiting run_agent.", "info")
    
    def execute(self, subtask: str):
        """
        控制机械臂执行, 退出 execute 函数的方式：
            1.is_post_run = True, 键盘中断 或是 捕获到某种异常
            2.subtask_executed_done = True, 子任务检查完成, 检查子任务是否完成由一个 llm 负责检查
        """
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        
        timestep = 0 
        self.log(f"Executing subtask: {subtask}", "info")

        with self.subtask_executed_done_lock:
            subtask_executed_done = self.subtask_executed_done

        # 只要检测到任务没有完成(由 llm 检测) 并且 is_post_run 没有被改成 True 的时候执行循环
        with self.post_run_lock:
            in_post_run = self.in_post_run
        while subtask_executed_done == False and in_post_run == False:
            
            # 如果大于时间步, 强制退出
            # TODO: 定义 max_subtask_timesteps, 经验判断一条任务可以给 600 个 steps , 允许三次失败(attempt <= 3)
            if timestep >= self.args.max_subtask_timesteps:
                self.log(f"Subtask execution reached max_timesteps ({self.args.max_subtask_timesteps}).", "warning")
                with self.post_run_lock:
                    self.in_post_run = True # Force stop
                break

            start_time = time.time()
            try:
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    with self.image_lock:

                        current_left_image = self.left_image
                        current_wrist_image = self.wrist_image
                        current_joint_pos = self.joint_position
                        current_gripper_pos = self.gripper_position

                    # TODO: 检查一下 image_tools.resize_with_pad 引入的延迟
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(current_left_image, 224, 224),
                        "observation/wrist_image": image_tools.resize_with_pad(current_wrist_image, 224, 224),
                        "observation/state": np.concatenate((current_joint_pos, current_gripper_pos), axis=0),
                        "prompt": subtask,
                    }

                    # 禁止键盘中断
                    with prevent_keyboard_interrupt():
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
                timestep += 1 

                elapsed_time = time.time() - start_time
                sleep_time = (1 / self.args.control_frequency) - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                with self.subtask_executed_done_lock:
                    subtask_executed_done = self.subtask_executed_done

            # Thinking: 是否需要键盘中断就全局暂停？停止整个任务？
            except KeyboardInterrupt:
                self.log("KeyboardInterrupt in execute loop. Stopping.", "info")
                with self.post_run_lock:
                    self.in_post_run = True # Gracefully stop
                break
            except Exception as e:
                self.log(f"\n[FATAL] Error in execute loop: {e}", "error")
                with self.post_run_lock:
                    self.in_post_run = True # Force stop on error
                break

    def log(self, message: str, message_type: str = "info"):
        """ (User-provided) A thread-safe logger. """
        with self.log_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_message = f"[{timestamp}] [{message_type.upper()}] {message}"
            
            print(log_message)
            
            if hasattr(self, 'log_file') and self.log_file and (self.log_file.closed == False):
                try:
                    self.log_file.write(log_message + "\n")
                    self.log_file.flush()
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [LOG_ERROR] Failed to write to log file: {e}")
    
    # def save_image(self, image, filename_prefix):
    #     """Saves an image to the current episode's image directory."""
    #     if image is None:
    #         self.log(f"Cannot save image {filename_prefix}, image is None.", "warning")
    #         return
            
    #     if hasattr(self, 'img_dir') == False or bool(self.img_dir) == False:
    #          self.log(f"Cannot save image, img_dir is not set (episode not initialized?).", "error")
    #          return
             
    #     try:
    #         img_path = os.path.join(self.img_dir, f"{filename_prefix}.png")
    #         Image.fromarray(image).save(img_path)
    #         self.log(f"Image saved to {img_path}", "debug")
    #     except Exception as e:
    #         self.log(f"Failed to save image {filename_prefix}: {e}", "error")

    def close_episode(self):
        """Cleans up resources at the end of an episode."""
        self.log("Closing episode.", "info")
        
        # # Release video writer
        # if hasattr(self, 'video_writer') and self.video_writer:
        #      self.video_writer.release()
        #      self.log("Video writer released.", "debug")
        #      self.video_writer = None

        # Close log file
        if hasattr(self, 'log_file') and self.log_file and (self.log_file.closed == False):
            self.log_file.close()
            self.log("Log file closed.", "debug")
            self.log_file = None

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