# 静态规划！静态：一次性生成所有子任务
# 修改版本：改进的日志系统，使用北京时间（UTC+8），精确到微秒

import dataclasses
import faulthandler
import os
import select
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional
import concurrent.futures
import json
import re
import concurrent.futures
import json

import numpy as np
import pandas as pd
from PIL import Image
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from droid.camera_utils.wrappers import multi_camera_wrapper as mcw
from droid.robot_env import RobotEnv
from ZEDCamera import ZedCamera

from utils import prevent_keyboard_interrupt, clear_input_buffer
from utils import parse_llm_output_to_list_regex, extract_boolean_answer, process_image

from pi05_main import _extract_observation

faulthandler.enable()

# 定义北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))
CROP_RATIOS = (0.27, 0.13)  # 与 pi05_main_async 对齐的左右裁剪比例


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


@contextmanager
def time_scope(name: str, log_file=None):
    """简单计时上下文，兼容原 timer.timer 接口的第二个参数。"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        msg = f"[TIMER] {name}: {duration:.4f}s"
        print(msg)
        if log_file is not None:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except Exception:
                pass


@dataclasses.dataclass
class Args:
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    external_camera: Optional[str] = "left"

    fetch_hz: float = 30.0
    first_frame_timeout_s: float = 5.0
    max_timesteps: int = 3600
    max_duration: int = 1800  # seconds

    open_loop_horizon: int = 18
    control_frequency: int = 50

    remote_host: str = "162.105.195.74"
    remote_port: int = 8000

    record_dir: str = "record"

    # TODO: 保存部署数据功能尚未实现
    save_deployment_data: bool = False

    max_subtask_timesteps: int = 1200

    use_env_camera: bool = False  # 仅支持禁用 env 相机；True 将抛错
    
    vllm_port: Optional[int] = None  # 如果为 None，则默认为 8000


class ManualCameraManager:
    """轻量级 ZED 相机管理器：禁用 env 相机时手动抓取外部与腕部相机。"""

    def __init__(self, args: Args):
        self.args = args
        self.external_serial = self._resolve_external_serial(args)
        self.wrist_serial = args.wrist_camera_id

        self.external_camera = self._init_camera(self.external_serial, "external") if self.external_serial else None
        self.wrist_camera = self._init_camera(self.wrist_serial, "wrist") if self.wrist_serial else None

    def _resolve_external_serial(self, args: Args) -> Optional[str]:
        if args.external_camera == "right":
            return args.right_camera_id
        return args.left_camera_id

    def _init_camera(self, serial: str | None, label: str):
        if not serial or serial == "<your_camera_id>":
            return None
        try:
            return ZedCamera(serial_number=int(serial))
        except Exception as exc:
            raise RuntimeError(f"初始化 {label} ZED 相机失败（serial={serial}）: {exc}") from exc

    def capture_images(self):
        images = {}
        if self.external_camera:
            ext_left, _ = self.external_camera.capture_frame()
            if ext_left is not None:
                images[f"{self.external_serial}_left"] = ext_left
        if self.wrist_camera:
            wrist_left, _ = self.wrist_camera.capture_frame()
            if wrist_left is not None:
                images[f"{self.wrist_serial}_left"] = wrist_left
        return images

    def close(self):
        if self.external_camera:
            try:
                self.external_camera.close()
            except Exception:
                pass
        if self.wrist_camera:
            try:
                self.wrist_camera.close()
            except Exception:
                pass


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
        self.image_thread = None
        self.image_thread_stop_event = threading.Event()
        self.manual_cam_mgr: Optional[ManualCameraManager] = None
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
        self.frame_buffer = []
        self.vlm_logs = []
        self.vlm_request_idx = 0
        self.episode_closed = False
        self.global_step = 0

        if self.args.use_env_camera:
            raise ValueError("Async mode only supports manual ZEDCamera capture; set use_env_camera=False.")

        # 禁用 RobotEnv 内置相机初始化，改为手动 ZED 抓取（与 pi05_main_async 一致）
        mcw.gather_zed_cameras = lambda: []
        self.log("Disabled env cameras; will use ZEDCamera manual capture.", "info")

        self.env = RobotEnv(action_space="joint_position", gripper_action_space="position")

        self.log("Created the droid env!", message_type="info")

        try:
            self.manual_cam_mgr = ManualCameraManager(self.args)
            self.log("ManualCameraManager initialized for ZED cameras.", "info")
        except Exception as e:
            self.log(f"Failed to initialize ManualCameraManager: {e}", "error")
            raise

        self.policy = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)
        self.log(f"Successfully connected to {self.args.remote_host}:{self.args.remote_port}!", "info")

        self.init_vlm()
        self.log("Init RoboBrain done", "info")

        self.init_threads()
        self.log("Init threads done", "info")
    
    def init_vlm(self):
        from scripts.robobrain import RoboBrain
        self.vlm = RoboBrain(8000 if self.args.vllm_port is None else self.args.vllm_port)

    def init_threads(self):
        # 后台持续抓取相机帧，避免在其他线程调用 get_observation 引发环境锁
        self.image_thread = threading.Thread(target=self.capture_images, daemon=True)
        self.image_thread.start()

        self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_listener_thread.start()
        
        self.check_subtask_success_thread = threading.Thread(target=self.check_subtask_success, daemon=True)
        self.check_subtask_success_thread.start()

        self.monitor_execution_time_thread = threading.Thread(target=self.monitor_execution_time, daemon=True)
        self.monitor_execution_time_thread.start()

    def capture_images(self):
        """
        背景线程：仅抓取相机图像并缓存，避免跨线程调用 env.get_observation 触发环境锁。
        机器人状态在主线程使用 env.get_state() 同步读取。
        """
        target_interval = 1.0 / self.args.fetch_hz if self.args.fetch_hz > 0 else 0.0
        
        while self.running and not self.image_thread_stop_event.is_set():
            start = time.time()
            try:
                if self.manual_cam_mgr is None:
                    raise RuntimeError("ManualCameraManager is not initialized.")
                image_observations = self.manual_cam_mgr.capture_images()

                with self.image_lock:
                    for key in image_observations:
                        if self.args.left_camera_id in key and "left" in key:
                            self.left_image = image_observations[key]
                        elif self.args.right_camera_id in key and "left" in key:
                            self.right_image = image_observations[key]
                        elif self.args.wrist_camera_id in key and "left" in key:
                            self.wrist_image = image_observations[key]
            except KeyboardInterrupt:
                self.image_thread_stop_event.set()
                break
            except Exception as e:
                self.log(f"Error in capture_images: {e}", "error")
                time.sleep(0.05)
                continue

            elapsed = time.time() - start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop_image_thread(self):
        self.image_thread_stop_event.set()
        if self.image_thread and self.image_thread.is_alive():
            self.image_thread.join(timeout=1.0)
        if self.manual_cam_mgr:
            try:
                self.manual_cam_mgr.close()
            except Exception:
                pass

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
                            current_image_raw = self.left_image
                            current_image = process_image(current_image_raw, crop_ratios=CROP_RATIOS)
                        
                        if current_image is None:
                            self.log("check_subtask_success: 图像尚未准备好, 跳过此次检查.", "warning")
                            time.sleep(1)
                            continue

                        # 获取 log_file（可能为 None）
                        log_file = self._get_log_file()
                        
                        # 执行 VLM 检查
                        self.log(f"[VLM] Starting subtask completion check for: '{current_subtask_local}'", "debug")
                        check_start = time.time()
                        vl_inputs = {
                            "images": {
                                "current_image": current_image
                            },
                            "user_prompt": current_subtask_local
                        }
                        prompt_text = self.vlm._build_prompt("subtask_complete_check", vl_inputs)
                        with time_scope("subtask complete check", log_file):
                            subtask_completed_str: str = self.vlm.request_task(
                                task_name="subtask_complete_check",
                                vl_inputs=vl_inputs
                            )
                        self._log_vlm_interaction(
                            "subtask_complete_check",
                            vl_inputs,
                            subtask_completed_str,
                            check_start,
                            time.time(),
                            prompt_text,
                        )
                        self._log_vlm_interaction(
                            "subtask_complete_check",
                            {"images": {"current_image": current_image}, "user_prompt": current_subtask_local},
                            subtask_completed_str,
                            check_start,
                            time.time(),
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
        vl_inputs = {
            "images": {
                "initial_image": self.initial_image,
            },
            "user_prompt": self.user_prompt
        }
        prompt_text = self.vlm._build_prompt("global_instruction_proposal", vl_inputs)
        
        with time_scope("global instruction proposal", log_file):
            global_task_str: str = self.vlm.request_task(
                task_name="global_instruction_proposal",
                vl_inputs=vl_inputs
            )
        self._log_vlm_interaction(
            "global_instruction_proposal",
            vl_inputs,
            global_task_str,
            proposal_start,
            time.time(),
            prompt_text,
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
            current_image = process_image(self.left_image, crop_ratios=CROP_RATIOS)

        log_file = self._get_log_file()
        
        self.log("[VLM] Checking global task completion...", "info")
        check_start = time.time()
        vl_inputs = {
            "images": {
                "initial_image": self.initial_image,
                "current_image": current_image
            },
            "user_prompt": self.user_prompt
        }
        prompt_text = self.vlm._build_prompt("global_task_complete_check", vl_inputs)
        
        with time_scope("global task complete check", log_file):
            task_completed_str: str = self.vlm.request_task(
                task_name="global_task_complete_check",
                vl_inputs=vl_inputs
            )
        self._log_vlm_interaction(
            "global_task_complete_check",
            vl_inputs,
            task_completed_str,
            check_start,
            time.time(),
            prompt_text,
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
            # 扩展用于帧和 VLM 图像的目录
            self.frames_dir = os.path.join(self.current_episode_dir, 'frames')
            os.makedirs(self.frames_dir, exist_ok=True)
            self.vlm_images_dir = os.path.join(self.current_episode_dir, 'vlm_images')
            os.makedirs(self.vlm_images_dir, exist_ok=True)

            # 重置缓冲
            self.frame_buffer = []
            self.vlm_logs = []
            self.vlm_request_idx = 0
            self.episode_closed = False
            self.global_step = 0

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

        # 等待后台抓取线程准备好首帧
        self.log("Waiting for camera to be ready...", "info")
        wait_start = time.time()

        while True:
            with self.image_lock:
                ready_image = self.left_image
            if ready_image is not None:
                break
            if time.time() - wait_start > self.args.first_frame_timeout_s:
                self.log(f"Camera timeout after {self.args.first_frame_timeout_s}s!", "error")
                return
            time.sleep(0.1)
        
        self.log(f"Camera ready after {time.time() - wait_start:.2f}s", "info")
        
        with self.image_lock:
            self.initial_image = process_image(self.left_image, crop_ratios=CROP_RATIOS)

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
        current_left_image = None
        current_wrist_image = None
        
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
                        raw_left_image = self.left_image
                        raw_wrist_image = self.wrist_image

                    if raw_left_image is None or raw_wrist_image is None:
                        self.log("[EXECUTE] Images not ready yet, waiting...", "warning")
                        time.sleep(0.05)
                        continue

                    try:
                        # 状态在主线程同步读取，避免跨线程触发环境锁
                        state_dict, _ = self.env.get_state()
                        raw_obs = {
                            "robot_state": state_dict,
                            "image": {
                                f"{self.args.left_camera_id}_left": raw_left_image,
                                f"{self.args.wrist_camera_id}_left": raw_wrist_image,
                            },
                        }
                        curr_obs = _extract_observation(self.args, raw_obs, crop_ratios=CROP_RATIOS)

                        current_left_image = curr_obs["left_image"]
                        current_wrist_image = curr_obs["wrist_image"]
                        current_joint_pos = curr_obs["joint_position"]
                        current_gripper_pos = curr_obs["gripper_position"]

                        self.joint_position = current_joint_pos
                        self.gripper_position = current_gripper_pos
                    except Exception as e:
                        self.log(f"[EXECUTE] build observation failed: {e}", "error")
                        time.sleep(0.05)
                        continue

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
                # 缓存当前帧，结束后统一写盘（每步都存）
                try:
                    if current_left_image is not None or current_wrist_image is not None:
                        self.frame_buffer.append(
                            {
                                "subtask": subtask,
                                "step": timestep,
                                "global_step": self.global_step,
                                "left_image": current_left_image.copy() if current_left_image is not None else None,
                                "wrist_image": current_wrist_image.copy() if current_wrist_image is not None else None,
                            }
                        )
                except Exception:
                    pass

                timestep += 1
                self.global_step += 1 

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
            mt_lower = message_type.lower()

            # ANSI 颜色
            color_map = {
                "debug": "\033[90m",
                "info": "\033[0m",
                "outcome": "\033[92m",
                "warning": "\033[93m",
                "error": "\033[91m",
                "subtask": "\033[95m",  # 紫色
            }

            color_prefix = color_map.get(mt_lower, "\033[0m")

            # 特殊判定：episode header、subtask 启动等用紫色
            if mt_lower == "info":
                if (
                    "Episode initialized at" in message
                    or "Beijing Time" in message
                    or "User prompt" in message
                    or message.strip().startswith("=")
                    or "Generated " in message
                    or "[SUBTASK" in message
                    or "Starting execution of" in message
                ):
                    color_prefix = color_map["subtask"]

            # outcome：未完成用黄，完成用绿
            if mt_lower == "outcome":
                if "NOT completed" in message or "⏳" in message:
                    color_prefix = color_map["warning"]
                if "COMPLETED" in message or "✅" in message:
                    color_prefix = color_map["outcome"]

            color_reset = "\033[0m"

            # 始终输出到控制台（带颜色）
            print(f"{color_prefix}{log_message}{color_reset}")
            
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

    def _log_vlm_interaction(self, task_name: str, vl_inputs: dict, response: str, start_ts: float, end_ts: float, prompt_text: str | None = None):
        """记录 VLM 请求/响应，图像暂存到内存，结束时统一落盘。"""
        self.vlm_request_idx += 1
        request_id = self.vlm_request_idx
        entry = {
            "request_id": request_id,
            "task_name": task_name,
            "start_time": start_ts,
            "end_time": end_ts,
            "duration_s": end_ts - start_ts,
            "inputs": {k: v for k, v in vl_inputs.items() if k != "images"} if isinstance(vl_inputs, dict) else {},
            "prompt": prompt_text,
            "images": [],
            "images_pending": [],
            "response": response,
        }
        images = vl_inputs.get("images", {}) if isinstance(vl_inputs, dict) else {}
        for key, img in images.items():
            if img is None:
                continue
            try:
                entry["images_pending"].append({"key": key, "image": img.copy()})
            except Exception:
                pass
        self.vlm_logs.append(entry)

    def close_episode(self):
        """清理 episode 资源"""
        if getattr(self, "episode_closed", False):
            return

        self.log("=" * 80, "info")
        self.log("Closing episode.", "info")
        self.log(f"End time (Beijing): {get_beijing_timestamp()}", "info")
        self.log("=" * 80, "info")

        # 使用线程池并行写盘，无损 PNG（低压缩加速）
        save_tasks = []
        max_workers = max(8, (os.cpu_count() or 8))

        def _save_png(img_arr, path):
            Image.fromarray(img_arr).save(path, compress_level=1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            # 落盘缓存的帧
            if getattr(self, "frame_buffer", None) and getattr(self, "frames_dir", None):
                for item in self.frame_buffer:
                    subtask_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", item.get("subtask", "subtask"))
                    step_idx = item.get("step", 0)
                    g_step = item.get("global_step", step_idx)
                    for key in ["left_image", "wrist_image"]:
                        img = item.get(key)
                        if img is None:
                            continue
                        img_path = os.path.join(self.frames_dir, f"{key}_step_{g_step}_{subtask_safe}.png")
                        save_tasks.append(pool.submit(_save_png, img, img_path))

            # 落盘 VLM 请求里的图像
            if getattr(self, "vlm_logs", None) and getattr(self, "vlm_images_dir", None):
                for entry in self.vlm_logs:
                    if entry.get("images"):
                        continue
                    paths = []
                    pending_images = entry.pop("images_pending", [])
                    for img_item in pending_images:
                        key = img_item.get("key", "image")
                        img = img_item.get("image")
                        if img is None:
                            continue
                        img_path = os.path.join(self.vlm_images_dir, f"vlm_request_image_{entry['request_id']}_{key}.png")
                        paths.append({"key": key, "path": img_path})
                        save_tasks.append(pool.submit(_save_png, img, img_path))
                    entry["images"] = paths

            for task in save_tasks:
                try:
                    task.result()
                except Exception as e:
                    self.log(f"Image save task failed: {e}", "warning")

        # 写 VLM 请求日志
        if getattr(self, "vlm_logs", None):
            try:
                log_path = os.path.join(self.current_episode_dir or ".", "vlm_requests.jsonl")
                with open(log_path, "w", encoding="utf-8") as f:
                    for entry in self.vlm_logs:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.log(f"VLM requests saved to {log_path}", "info")
                # 可读性更好的版本（缩进），便于人工查看
                pretty_path = os.path.join(self.current_episode_dir or ".", "vlm_requests_readable.json")
                with open(pretty_path, "w", encoding="utf-8") as f:
                    json.dump(self.vlm_logs, f, ensure_ascii=False, indent=2)
                self.log(f"VLM requests (readable) saved to {pretty_path}", "info")
            except Exception as e:
                self.log(f"Failed to save VLM logs: {e}", "warning")

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

        self.episode_closed = True


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
        agent.stop_image_thread()
        agent.close_episode()
        agent.log("RoboAgent shutdown complete.", "info")
        sys.exit(0)
