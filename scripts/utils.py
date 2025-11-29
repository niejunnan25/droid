# ruff: noqa

import contextlib
import concurrent.futures
import datetime
import faulthandler
import os
import pickle
import re
import select
import signal
import sys
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from openpi_client import image_tools

faulthandler.enable()


KNOWN_OBJECTS = ["corn", "green pepper", "red pepper", "garlic", "potato", "cabbage"]


class TrajectoryCollector:
    
    """Collects and saves rollout data (observations, actions, metadata, optional video)."""
    
    def __init__(
        self,
        save_dir: str = "collected_rollouts",
        *,
        enable_rollout: bool = True,
        enable_video: bool = False,
        video_save_dir: str | None = None,
        external_camera: str = "left",
        video_cameras: list[str] | None = None,
        write: bool = False,
        video_executor: concurrent.futures.Executor | None = None,
    ):
        self.save_dir = save_dir
        self.enable_rollout = enable_rollout
        self.enable_video = enable_video
        self.video_save_dir = video_save_dir
        self.external_camera = external_camera
        # Cameras to dump as video; defaults to external camera + wrist
        self.video_camera_keys = video_cameras if video_cameras is not None else [
            f"{self.external_camera}_image",
            "wrist_image",
        ]
        # 控制是否在保存的帧上叠加文本
        self.write_overlay = write
        self.video_executor = video_executor

        self.trajectory: dict[str, list | dict] = {}
        self.video_frames: dict[str, list[np.ndarray]] | None = None
        self.episode_slug: str | None = None
        self.overlay_info: list[dict] = []

        if self.enable_rollout:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.enable_video and self.video_save_dir:
            os.makedirs(self.video_save_dir, exist_ok=True)
    
    def start_episode(self, instruction: str, *, episode_slug: str | None = None):
        """Initialize a new episode."""
        self.episode_slug = episode_slug or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_frames = {key: [] for key in self.video_camera_keys} if self.enable_video else None
        self.overlay_info = []

        self.trajectory = {
            "observations": [],
            "actions": [],
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "instruction": instruction,
                "num_steps": 0,
            },
        }
    
    def _draw_overlay(self, frame: np.ndarray, *, step_idx: int, action: np.ndarray, chunk_idx: int | None, chunk_size: int | None):
        """在帧左上角叠加时间步、chunk 进度和动作。"""
        try:
            display_action = np.array2string(action, precision=3, separator=",", suppress_small=True)
        except Exception:
            display_action = str(action)

        texts = [f"t={step_idx}"]
        if chunk_idx is not None and chunk_size is not None and chunk_size > 0:
            texts.append(f"{chunk_idx}/{chunk_size}")
        texts.append(f"action: {display_action}")

        # cv2 期望 BGR
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # 黑色文字
        thickness = 2
        line_height = 40

        for i, t in enumerate(texts):
            cv2.putText(img_bgr, t, (x, y + i * line_height), font, font_scale, color, thickness, cv2.LINE_AA)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def add_step(self, observation: dict, action: np.ndarray, *, step_idx: int | None = None, chunk_idx: int | None = None, chunk_size: int | None = None):
        """Add an observation-action pair (and optional video frame)."""
        if self.enable_rollout:
            self.trajectory["observations"].append(observation.copy())
            self.trajectory["actions"].append(action.copy())
        self.overlay_info.append(
            {
                "step_idx": step_idx if step_idx is not None else len(self.overlay_info),
                "action": action.copy(),
                "chunk_idx": chunk_idx,
                "chunk_size": chunk_size,
            }
        )

        if self.enable_video and self.video_frames is not None:
            info = self.overlay_info[-1]
            for key in self.video_frames:
                frame = observation.get(key)
                if frame is not None:
                    if self.write_overlay:
                        frame_to_store = self._draw_overlay(
                            frame.copy(),
                            step_idx=info["step_idx"],
                            action=info["action"],
                            chunk_idx=info["chunk_idx"],
                            chunk_size=info["chunk_size"],
                        )
                    else:
                        frame_to_store = frame
                    self.video_frames[key].append(frame_to_store)
    
    def save_episode(self, success: bool = True, completed_steps: int | None = None, *, episode_slug: str | None = None):
        """Save the entire trajectory and/or video to disk."""
        slug = episode_slug or self.episode_slug or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir: str | None = None
        images_dir: str | None = None
        pickle_path: str | None = None
        num_steps: int = 0
        saved_images = 0
        observations = self.trajectory.get("observations", [])

        if self.enable_rollout and observations:
            episode_dir = os.path.join(self.save_dir, f"episode_{slug}")
            os.makedirs(episode_dir, exist_ok=True)

            num_steps = len(observations)
            self.trajectory["metadata"]["num_steps"] = num_steps
            if completed_steps is not None:
                self.trajectory["metadata"]["completed_steps"] = int(completed_steps)
                self.trajectory["metadata"]["interrupted"] = completed_steps < num_steps
            else:
                self.trajectory["metadata"]["completed_steps"] = num_steps
                self.trajectory["metadata"]["interrupted"] = False
            self.trajectory["metadata"]["success"] = success

            images_dir = os.path.join(episode_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            trajectory_to_save = {
                "metadata": self.trajectory["metadata"],
                "actions": np.array(self.trajectory["actions"]),
                "observations": [
                    {k: v for k, v in obs.items() if not k.endswith("_image")}
                    for obs in observations
                ],
            }

            pickle_path = os.path.join(episode_dir, "trajectory.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(trajectory_to_save, f)

        if self.enable_video and self.video_frames:
            if not self.video_save_dir:
                print("⚠️ 视频未保存：video_save_dir 未设置")
                return
            for key, frames in self.video_frames.items():
                if not frames:
                    continue
                cam_label = key.replace("_image", "")
                filename = f"episode_{slug}"
                # Only append label if multiple cameras to avoid breaking old single-cam naming
                if len(self.video_frames) > 1:
                    filename += f"_{cam_label}"
                filename += ".mp4"
                video_path = os.path.join(self.video_save_dir, filename)
                if self.video_executor:
                    self.video_executor.submit(save_video, frames, video_path, 30)
                else:
                    save_video(frames, video_path, fps=30)

        if images_dir:
            # 写图片可以并行加速，避免阻塞主线程
            save_tasks: list[tuple[str, np.ndarray]] = []
            for step_idx, obs in enumerate(observations):
                info = self.overlay_info[step_idx] if step_idx < len(self.overlay_info) else None
                for cam_name in ["left_image", "wrist_image"]:
                    img = obs.get(cam_name)
                    if img is None:
                        continue
                    img_np = img.astype(np.uint8)
                    if self.write_overlay and info is not None:
                        img_np = self._draw_overlay(
                            img_np,
                            step_idx=info["step_idx"],
                            action=info["action"],
                            chunk_idx=info["chunk_idx"],
                            chunk_size=info["chunk_size"],
                        )
                    img_path = os.path.join(images_dir, f"{cam_name}_{step_idx:04d}.jpg")
                    save_tasks.append((img_path, img_np))

            if save_tasks:
                max_workers = min(8, max(2, os.cpu_count() or 2))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self._save_single_image, path, frame)
                        for (path, frame) in save_tasks
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()
            saved_images = len(save_tasks)
            print(f"  - Saved {saved_images} images in parallel to {images_dir}")

        # Summaries should be printed after all saving work completes
        if episode_dir:
            print(f"\n✓ Saved trajectory to {episode_dir}")
            print(f"  - Steps: {num_steps}")
            print(f"  - Success: {success}")
            print(f"  - Images dir: {images_dir}")
            print(f"  - Images saved: {saved_images}")
            print(f"  - Metadata + actions: {pickle_path}\n")

    @staticmethod
    def _save_single_image(path: str, frame: np.ndarray):
        """Save a single RGB frame to disk as JPG."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # cv2 expects BGR; also handle grayscale safely
        if img.ndim == 3 and img.shape[2] == 3:
            img_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_write = img

        if not cv2.imwrite(path, img_to_write):
            print(f"Failed to save image to {path}")



def clear_input_buffer():
    # Clear input buffer using different methods for better compatibility
    try:
        # For Windows
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        # For Unix/Linux/MacOS
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        # Fallback method
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)



@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt



def parse_llm_output_to_list_regex(llm_output: str) -> list[str]:
    """
    使用正则表达式将LLM返回的带项目符号的列表解析为Python列表。
    
    这个模式会查找以 "- " 开头，然后捕获该行余下所有内容的文本。
    """
    
    # 模式 r"-\s+(.*)" 的解释:
    # -   : 匹配一个字面上的连字符 (hyphen)
    # \s+ : 匹配一个或多个空白字符 (包括空格、制表符等)
    # (.*): 捕获 (capture group) 该行上随后的所有字符
    pattern = r"-\s+(.*)"
    
    # re.findall 会找到所有匹配项，并只返回捕获组 (.*) 中的内容
    steps = re.findall(pattern, llm_output)
    
    # 额外步骤：去除捕获内容两端可能存在的空白
    return [step.strip() for step in steps]


def extract_boolean_answer(text: str) -> bool | None:
    """
    从字符串开头提取 True 或 False（忽略大小写）。
    """
    # 模式解释：
    # ^       : 匹配字符串的开头
    # \s* : 匹配零个或多个空格（允许字符串开头有空格）
    # (True|False): 捕获组，匹配 'True' 或 'False'
    # re.IGNORECASE: 忽略大小写，因此 'true', 'TRUE', 'True' 都能匹配
    pattern = r"^\s*(True|False)"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # match.group(1) 提取捕获组的内容 ('True' 或 'False')
        # 然后转换为小写并检查是否为 'true'
        return match.group(1).lower() == 'true'
    else:
        # 如果没有匹配到，返回 None 或者抛出异常，取决于你的需求
        return None



def process_image(img, crop_ratios=None):
    """(User-provided) Processes raw image: BGR -> RGB and optional crop."""
    if img is None:
        return None
    img = img[..., :3][..., ::-1]  # Assume BGR, convert to RGB
    pil_img = Image.fromarray(img)
    if crop_ratios:
        pil_img = crop_left_right(pil_img, *crop_ratios)
    return np.array(pil_img)


def crop_left_right(image: Image.Image, left_ratio: float, right_ratio: float) -> Image.Image:
    """
    按比例裁剪左右两边。
    left_ratio: 左边裁掉的比例（0~1）
    right_ratio: 右边裁掉的比例（0~1）
    """
    width, height = image.size
    left = int(width * left_ratio)
    right = int(width * (1 - right_ratio))
    return image.crop((left, 0, right, height))


def _extract_observation(args, obs_dict, *, save_to_disk=False, crop_ratios=None):
    """Extracts images and robot state, with optional left/right cropping."""

    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]
    # 这个地方拿到的是原始的相机 left_image, wrist_image
    left_image = process_image(left_image, crop_ratios=crop_ratios)
    right_image = process_image(right_image, crop_ratios=crop_ratios)
    wrist_image = process_image(wrist_image, crop_ratios=crop_ratios)

    # 机器人状态
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        images_to_save = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if images_to_save:
            combined_image = np.concatenate(images_to_save, axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def save_video(frames, filename, fps: int = 30):
    """
    Save a list of RGB frames to an MP4 file using OpenCV at the given fps.
    Uses mp4v for broad compatibility and speed.
    """
    if not frames:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Infer size from first valid frame
    first = next((f for f in frames if f is not None), None)
    if first is None or first.ndim != 3 or first.shape[2] != 3:
        print("视频保存失败：首帧格式不符合期望 HxWx3")
        return
    height, width, _ = first.shape

    # Try codecs in order; fall back to avi/MJPG if mp4 fails
    candidates = []
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".mp4":
        candidates.append(("mp4v", filename))
        candidates.append(("avc1", filename))
    candidates.append(("MJPG", base + ".avi"))

    writer = None
    chosen_path = None
    chosen_codec = None
    for codec, path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            chosen_path = path
            chosen_codec = codec
            break
        writer.release()
        writer = None

    if writer is None:
        print("视频保存失败：无法打开 VideoWriter，可能缺少编码器 (mp4/avi)")
        return

    for frame in frames:
        if frame is None:
            continue
        # Ensure uint8 and RGB shape
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"视频保存完成 {chosen_path} (codec={chosen_codec}, {width}x{height}@{fps}fps)")



def save_actions_to_pickle(executed_actions, filename):
    """
    Saves the list of executed actions and the corresponding instruction to a pickle file.

    Args:
        executed_actions (list): A list of numpy arrays, where each array is an action.
        filename (str): The path to the output .pkl file.
    """
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    # Convert list of actions to a single numpy array
    actions_array = np.array(executed_actions)

    # Store data in a dictionary for clarity
    data_to_save = {
        'actions': actions_array,
    }

    # Write to a pickle file in binary mode
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"✅ Actions successfully saved to {filename}")


def extract_object(instruction: str) -> Optional[str]:
    text = instruction.lower()

    for obj in KNOWN_OBJECTS:
        # 用 \b 确保是独立的单词，避免 "scorn" 里误匹配 "corn"
        pattern = r"\b" + re.escape(obj) + r"\b"
        if re.search(pattern, text):
            return obj
    return None

# 测试代码
if __name__ == "__main__":
    img_path = "/home/ubuntu/pi0/data_300/pick_up_the_cabbage_on_the_plate/0803_181934/image/left_20250803_181943_231061.png"  # 原图路径
    img = Image.open(img_path)

    # 自定义左右裁剪比例（例如左裁 0.2，右裁 0.2）
    cropped_img = crop_left_right(img, left_ratio=0.27, right_ratio=0.13)

    # 再缩放到 224x224
    resized_arr = image_tools.resize_with_pad(np.array(cropped_img), 224, 224)
    resized_img = Image.fromarray(resized_arr)  # 转回 PIL Image 才能保存

    # 保存
    cropped_img.save("cropped.jpg")
    resized_img.save("output_224.jpg")
    # resized_img = cropped_img.resize((224, 224), Image.LANCZOS)

    print(f"原图尺寸: {img.size}")
    print(f"裁剪后尺寸: {cropped_img.size}")
    print("已保存 cropped.jpg（裁剪后） 和 output_224.jpg（缩放到224x224）")
