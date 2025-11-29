# 静态规划！静态：一次性生成所有子任务
# 修改版本：改进的日志系统，使用北京时间（UTC+8），精确到微秒

import dataclasses
import faulthandler
import os
import select
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from droid.robot_env import RobotEnv

import timer
from utils import prevent_keyboard_interrupt, clear_input_buffer
from utils import crop_left_right, parse_llm_output_to_list_regex, extract_boolean_answer, process_image

faulthandler.enable()

# 定义北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


def get_beijing_timestamp() -> str:
    """
    获取当前北京时间的时间戳，精确到微秒
    格式: YYYY-MM-DD HH:MM:SS.ffffff
    """
    now = datetime.now(BEIJING_TZ)
    return now.strftime("%Y-%m-%d %H:%M:%S.%f")


def get_beijing_timestamp_short() -> str:
    """
    获取当前北京时间的短时间戳，精确到微秒
    格式: HH:MM:SS.ffffff
    """
    now = datetime.now(BEIJING_TZ)
    return now.strftime("%H:%M:%S.%f")


@dataclasses.dataclass
class Args:
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    external_camera: Optional[str] = "left"
    max_timesteps: int = 3600
    max_duration: int = 1800  # seconds
    open_loop_horizon: int = 18
    control_frequency: int = 50
    remote_host: str = "162.105.195.74"
    remote_port: int = 8000

    record_dir: str = "record"
    save_deployment_data: bool = False

    action_space: str = "joint_position"
    gripper_action_space: str = "position"

    max_subtask_timesteps: int = 1200


class RoboAgent:
    def __init__(self, args):
        
        self.args = args
        self.log_lock = threading.Lock()

        # 初始化 log_file 为 None，避免线程访问时报错
        self.log_file = None
        self.log_file_path = None
        self.img_dir = None
        self.current_episode_dir = None

        self.left_image = None
        self.right_image = None
        self.wrist_image = None

        self.image_lock = threading.Lock()
        self.current_subtask_lock = threading.Lock()
        self.subtask_executed_done_lock = threading.Lock()
        self.post_run_lock = threading.Lock()

        self.current_subtask = None

        self.joint_position = None
        self.gripper_position = None
        self.running = True

        self.in_post_run = True  # 设为True，防止线程在 agent 准备好前错误地运行
        self.episode_start = time.time()

        self.start_time = time.time()
        self.consecutive_failed_attempts = 0
        self.subtask_executed_done = False

        self.env = RobotEnv(action_space=self.args.action_space, gripper_action_space=self.args.gripper_action_space)
        self.log("Created the droid env!", message_type="info")

        self.policy = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)
        self.log(f"Successfully connected to {self.args.remote_host}:{self.args.remote_port}!", "info")

        self.init_vlm()
        self.log("Init RoboBrain done", "info")

        print(self.env.get_observation())

        self.init_threads()
        self.log("Init threads done", "info")
    
    def init_vlm(self):
        from scripts.robobrain import RoboBrain
        self.vlm = RoboBrain(8002)

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
                print("capture_images: fetching observation...")
                obs = self.env.get_observation()
                image_observations = obs["image"]
                print(obs)
                print(f"capture_images: got images keys: {list(image_observations.keys())}")
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
                # 先读取状态，避免在锁内 sleep
                with self.post_run_lock:
                    is_post_run = self.in_post_run
                
                if is_post_run == False:
                    # select 有 0.1s 超时，这里不需要额外 sleep
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        sys.stdin.readline()
                        self.log("<Enter> detected, ending this episode.", message_type="info")
                        with self.post_run_lock:
                            self.in_post_run = True
                else:
                    time.sleep(0.1)  # 在锁外 sleep
                    
            except Exception as e:
                self.log(f"Error in keyboard_listener: {e}", "error")

            if self.running == False:
                break
    
    def monitor_execution_time(self):
        """Thread to monitor task execution time"""
        while True:
            try:
                time.sleep(1.0) 
                with self.post_run_lock:
                    if self.in_post_run == False:
                        elapsed_time = time.time() - self.episode_start
                        if elapsed_time > self.args.max_duration:
                            self.log(f"Episode exceeded maximum duration of {self.args.max_duration}s", "warning")
                            self.in_post_run = True

            except Exception as e:
                self.log(f"Error in monitor_execution_time: {e}", "error")
            
            if self.running == False:
                break

    def check_subtask_success(self):
        """线程函数：异步检查子任务是否完成"""
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
                            continue

                        # 获取 log_file（可能为 None）
                        log_file = self._get_log_file()
                        
                        # 执行 VLM 检查
                        self.log(f"[VLM] Starting subtask completion check for: '{current_subtask_local}'", "debug")
                        check_start = time.time()
                        
                        with timer.timer("subtask complete check", log_file):
                            subtask_completed_str: str = self.vlm.request_task(
                                task_name="subtask_complete_check",
                                vl_inputs={
                                    "images": {
                                        "current_image": current_image
                                    },
                                    "user_prompt": current_subtask_local
                                }
                            )
                        
                        check_duration = time.time() - check_start
                        self.log(f"[VLM] Subtask check completed in {check_duration:.4f}s, response: {subtask_completed_str[:100]}...", "debug")

                        with self.subtask_executed_done_lock:
                            self.subtask_executed_done = extract_boolean_answer(subtask_completed_str)
                            subtask_executed_done = self.subtask_executed_done
                        
                        if subtask_executed_done is None:
                            self.log("Error, subtask checking did not output in the expected format", "error")
                        elif subtask_executed_done:
                            self.log(f"✅ Subtask '{current_subtask_local}' is COMPLETED.", message_type="outcome")
                        else:
                            self.log(f"⏳ Subtask '{current_subtask_local}' is NOT completed, continue!", message_type="outcome")
            
            except Exception as e:
                self.log(f"Error in check_subtask_success: {e}", "error")

            time.sleep(3)

            if self.running == False:
                break

    def _get_log_file(self):
        """安全地获取 log_file，如果未初始化则返回 None"""
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            return self.log_file
        return None

    def get_global_instruction(self) -> list:
        """VLM 生成全局子任务列表"""
        log_file = self._get_log_file()
        
        self.log("[VLM] Requesting global instruction proposal...", "info")
        proposal_start = time.time()
        
        with timer.timer("global instruction proposal", log_file):
            global_task_str: str = self.vlm.request_task(
                task_name="global_instruction_proposal",
                vl_inputs={
                    "images": {
                        "initial_image": self.initial_image,
                    },
                    "user_prompt": self.user_prompt
                }
            )
        
        proposal_duration = time.time() - proposal_start
        self.log(f"[VLM] Global instruction proposal completed in {proposal_duration:.4f}s", "info")
        self.log(f"[VLM] Raw response: {global_task_str}", message_type="debug")
        
        subtask_list: list = parse_llm_output_to_list_regex(global_task_str)
        self.log(f"[VLM] Parsed {len(subtask_list)} subtasks: {subtask_list}", message_type="outcome")
        return subtask_list

    def check_global_complete(self) -> bool:
        """VLM 检查全局任务是否完成"""
        with self.image_lock:
            current_image = self.left_image

        log_file = self._get_log_file()
        
        self.log("[VLM] Checking global task completion...", "info")
        check_start = time.time()
        
        with timer.timer("global task complete check", log_file):
            task_completed_str: str = self.vlm.request_task(
                task_name="global_task_complete_check",
                vl_inputs={
                    "images": {
                        "initial_image": self.initial_image,
                        "current_image": current_image
                    },
                    "user_prompt": self.user_prompt
                }
            )
        
        check_duration = time.time() - check_start
        self.log(f"[VLM] Global check completed in {check_duration:.4f}s, response: {task_completed_str[:100]}...", "debug")
        
        task_completed = extract_boolean_answer(task_completed_str)

        if task_completed is None:
            self.log("Error, global checking did not output in the expected format.", "error")
            return False
            
        if task_completed:
            self.log(f"✅ Global task '{self.user_prompt}' is COMPLETED!", message_type="outcome")
        else:
            self.log(f"⏳ Global task '{self.user_prompt}' is NOT completed, continuing...", message_type="outcome")
            
        return task_completed

    ################################################ 执行函数 (Execution Functions) ###############################################
    
    def init_episode(self):
        """初始化一个新的 episode：创建日志目录和日志文件"""
        try:
            # 使用北京时间创建时间戳
            beijing_now = datetime.now(BEIJING_TZ)
            timestamp = beijing_now.strftime("%Y%m%d_%H%M%S")
            
            self.current_episode_dir = os.path.join(self.args.record_dir, timestamp)
            os.makedirs(self.current_episode_dir, exist_ok=True)

            # 创建日志文件
            self.log_file_path = os.path.join(self.current_episode_dir, 'run.log')
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            
            # 创建 images 文件夹
            self.img_dir = os.path.join(self.current_episode_dir, 'images')
            os.makedirs(self.img_dir, exist_ok=True)

            # 让 VLM 也使用同一个日志系统
            self.vlm.set_logging(self.log_file, self.img_dir)

            # 写入 episode 头信息
            self.log("=" * 80, "info")
            self.log(f"Episode initialized at: {self.current_episode_dir}", "info")
            self.log(f"Beijing Time: {beijing_now.strftime('%Y-%m-%d %H:%M:%S.%f')}", "info")
            self.log(f"User prompt: {self.user_prompt}", "info")
            self.log("=" * 80, "info")
                
        except Exception as e:
            self.log(f"Error in init_episode: {e}", "error")
            self.running = False 

    def reset_flags(self):
        """重置 episode 相关的标志位"""
        self.log("Resetting episode flags.", "debug")
        with self.post_run_lock:
            self.in_post_run = False
        with self.subtask_executed_done_lock:
            self.subtask_executed_done = False
        self.episode_start = time.time()
        self.consecutive_failed_attempts = 0

    def run_agent(self):
        """主执行函数"""
        self.user_prompt = input("Please enter your instruction\n>>>  ")

        # 等待相机准备好
        self.log("Waiting for camera to be ready...", "info")
        wait_start = time.time()

        while self.left_image is None:
            time.sleep(0.1)
            if time.time() - wait_start > 10:
                self.log("Camera timeout after 10s!", "error")
                return
        
        self.log(f"Camera ready after {time.time() - wait_start:.2f}s", "info")
        
        with self.image_lock:
            self.initial_image = self.left_image.copy()

        if self.initial_image is None:
            self.log("Failed to get initial_image from camera. Exiting.", "error")
            return

        self.log(f"User prompt received: {self.user_prompt}", message_type="outcome")

        while True:  # Outer loop (for entire task)

            self.time_step = 0
            self.init_episode()

            # VLM 生成子任务列表
            subtask_list: list = self.get_global_instruction()
            self.log(f"Generated {len(subtask_list)} subtasks for execution", "info")

            subtask_index = 0

            while True:  # Inner loop (for sub-tasks)
                
                # Step1: 从 subtask_list 获取当前需要执行的子任务
                with self.current_subtask_lock:
                    try:
                        self.current_subtask = subtask_list[subtask_index]
                        current_subtask = self.current_subtask
                    except IndexError:
                        self.log("All subtasks completed (list exhausted)", "info")
                        with self.post_run_lock:
                            self.in_post_run = True
                        break

                self.log("-" * 60, "info")
                self.log(f"[SUBTASK {subtask_index + 1}/{len(subtask_list)}] Starting: {current_subtask}", "info")
                self.log("-" * 60, "info")

                # Step2: Reset 与任务相关的标志位
                self.reset_flags()

                # 清空终端缓冲区
                clear_input_buffer()

                # Step3：执行
                execute_start = time.time()
                self.execute(current_subtask)
                execute_duration = time.time() - execute_start
                self.log(f"[SUBTASK {subtask_index + 1}] Execution took {execute_duration:.2f}s", "info")
                
                # Step4: 检查全局指令是否完成
                global_task_completed: bool = self.check_global_complete()

                if global_task_completed:
                    self.log("🎉 Global task completed!", "outcome")
                    break

                subtask_index += 1
            
            self.close_episode()
            self.log("Task finished, exiting run_agent.", "info")
            break
    
    def execute(self, subtask: str):
        """
        控制机械臂执行子任务
        退出条件：
            1. in_post_run = True (键盘中断或异常)
            2. subtask_executed_done = True (VLM 判断子任务完成)
            3. 超过 max_subtask_timesteps
        """
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        
        timestep = 0 
        self.log(f"[EXECUTE] Starting execution of: {subtask}", "info")

        with self.subtask_executed_done_lock:
            subtask_executed_done = self.subtask_executed_done

        with self.post_run_lock:
            in_post_run = self.in_post_run

        chunk_count = 0
        
        while subtask_executed_done == False and in_post_run == False:
            
            if timestep >= self.args.max_subtask_timesteps:
                self.log(f"[EXECUTE] ⚠️ Reached max_timesteps ({self.args.max_subtask_timesteps}), stopping.", "warning")
                with self.post_run_lock:
                    self.in_post_run = True
                break

            start_time = time.time()
            try:
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    chunk_count += 1

                    with self.image_lock:
                        current_left_image = self.left_image
                        current_wrist_image = self.wrist_image
                        current_joint_pos = self.joint_position
                        current_gripper_pos = self.gripper_position

                    request_data = {
                        "observation/image": image_tools.resize_with_pad(current_left_image, 224, 224),
                        "observation/wrist_image": image_tools.resize_with_pad(current_wrist_image, 224, 224),
                        "observation/state": np.concatenate((current_joint_pos, current_gripper_pos), axis=0),
                        "prompt": subtask,
                    }

                    with prevent_keyboard_interrupt():
                        inference_start = time.time()
                        pred_action_chunk = self.policy.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]
                        inference_time = time.time() - inference_start
                        
                        # 每个 chunk 开始时记录
                        self.log(f"[VLA] Chunk {chunk_count}: inference={inference_time:.4f}s, timestep={timestep}", "debug")

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Gripper action thresholding
                if action[-1].item() > 0.20:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                self.env.step(action)
                timestep += 1 

                # 每 50 步记录一次进度
                if timestep % 50 == 0:
                    self.log(f"[EXECUTE] Progress: timestep={timestep}, chunks={chunk_count}", "info")

                elapsed_time = time.time() - start_time
                sleep_time = (1 / self.args.control_frequency) - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 更新循环条件变量
                with self.subtask_executed_done_lock:
                    subtask_executed_done = self.subtask_executed_done
                with self.post_run_lock:
                    in_post_run = self.in_post_run

            except KeyboardInterrupt:
                self.log("[EXECUTE] ⛔ KeyboardInterrupt detected, stopping.", "info")
                with self.post_run_lock:
                    self.in_post_run = True
                break
            except Exception as e:
                self.log(f"[EXECUTE] ❌ FATAL Error: {e}", "error")
                with self.post_run_lock:
                    self.in_post_run = True
                break

        # 记录子任务结束状态
        status = "✅ DONE" if subtask_executed_done else ("⛔ INTERRUPTED" if in_post_run else "❓ UNKNOWN")
        self.log(f"[EXECUTE] Finished after {timestep} timesteps ({chunk_count} chunks). Status: {status}", "info")

    def log(self, message: str, message_type: str = "info"):
        """
        线程安全的日志函数，同时输出到控制台和文件
        使用北京时间（UTC+8），精确到微秒
        """
        with self.log_lock:
            # 获取北京时间，精确到微秒
            timestamp = get_beijing_timestamp()
            log_message = f"[{timestamp}] [{message_type.upper():8}] {message}"
            
            # 始终输出到控制台
            print(log_message)
            
            # 如果日志文件已打开，同时写入文件
            if self.log_file is not None and not self.log_file.closed:
                try:
                    self.log_file.write(log_message + "\n")
                    self.log_file.flush()
                except Exception as e:
                    print(f"[{get_beijing_timestamp_short()}] [LOG_ERROR] Failed to write to log file: {e}")

    def save_image(self, image, filename_prefix: str):
        """保存图像到当前 episode 的 images 目录"""
        if image is None:
            self.log(f"Cannot save image {filename_prefix}, image is None.", "warning")
            return
            
        if self.img_dir is None:
            self.log("Cannot save image, img_dir is not set (episode not initialized?).", "error")
            return
             
        try:
            # 使用北京时间作为文件名的一部分
            beijing_ts = datetime.now(BEIJING_TZ).strftime("%H%M%S_%f")
            img_path = os.path.join(self.img_dir, f"{filename_prefix}_{beijing_ts}.png")
            Image.fromarray(image).save(img_path)
            self.log(f"Image saved to {img_path}", "debug")
        except Exception as e:
            self.log(f"Failed to save image {filename_prefix}: {e}", "error")

    def close_episode(self):
        """清理 episode 资源"""
        self.log("=" * 80, "info")
        self.log("Closing episode.", "info")
        self.log(f"End time (Beijing): {get_beijing_timestamp()}", "info")
        self.log("=" * 80, "info")

        # 关闭日志文件
        if self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.flush()
                self.log_file.close()
                print(f"[{get_beijing_timestamp_short()}] [INFO    ] Log file closed: {self.log_file_path}")
            except Exception as e:
                print(f"[{get_beijing_timestamp_short()}] [ERROR   ] Failed to close log file: {e}")
            finally:
                self.log_file = None


if __name__ == '__main__':
    # 1. Parse arguments
    args = tyro.cli(Args)
    
    # 2. Create the agent
    agent = RoboAgent(args)
    
    # 3. Run the agent
    try:
        agent.run_agent()
    except KeyboardInterrupt:
        agent.log("Shutdown requested by user.", "info")
    except Exception as e:
        agent.log(f"An uncaught exception occurred: {e}", "error")
    finally:
        agent.running = False
        agent.close_episode()
        agent.log("RoboAgent shutdown complete.", "info")
        sys.exit(0)

