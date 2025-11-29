# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional, Union
import cv2  # Make sure OpenCV is imported
from utils import crop_left_right
faulthandler.enable()
import pickle

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

CROP_RATIOS = (0.27, 0.13)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36276705"  # e.g., "24259877"
    right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    wrist_camera_id: str = "13132609"  # e.g., "13062452"

    # Policy parameters
    external_camera: Optional[str] = (
        "left"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )
    # Rollout parameters
    max_timesteps: int = 1200
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host : str = "162.105.195.74"
    # remote_host: str = "localhost"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
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


# ===============================================================================
# MODIFIED HELPER FUNCTION TO ADD TEXT TO VIDEO FRAMES (VERSION 2)
# ===============================================================================
def add_info_to_frame(
    frame: np.ndarray,
    step: int,
    chunk_count: int,
    full_action_chunk: np.ndarray,
    current_action_index: int,
    current_action: np.ndarray,
    next_action: Optional[np.ndarray],
    instruction: str,
) -> np.ndarray:
    """
    Adds detailed text information to a video frame's top-left corner.

    Args:
        frame (np.ndarray): The input image frame (in RGB format).
        step (int): The current timestep.
        chunk_count (int): The count of how many chunks have been inferred.
        full_action_chunk (np.ndarray): The entire predicted action chunk array.
        current_action_index (int): The zero-based index of the current action in the chunk.
        current_action (np.ndarray): The action currently being executed.
        next_action (Optional[np.ndarray]): The next action to be executed. Can be None.
        instruction (str): The language instruction.

    Returns:
        np.ndarray: The frame with the text overlay, in RGB format.
    """
    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    # --- Text Properties ---
    # Larger font for main info
    main_font = cv2.FONT_HERSHEY_SIMPLEX
    main_font_scale = 0.6
    main_font_color = (0, 0, 0)  # Black in BGR
    main_thickness = 2
    line_type = cv2.LINE_AA

    # Smaller font for the action chunk array to save space
    chunk_font = cv2.FONT_HERSHEY_SIMPLEX
    chunk_font_scale = 0.4
    chunk_font_color = (0, 0, 0)
    chunk_thickness = 1
    
    # --- Text Content Generation ---
    texts_to_draw = []
    
    # 1. Steps
    texts_to_draw.append(f"1. Steps: {step}")
    
    # 2. Chunk Count
    texts_to_draw.append(f"2. Chunk Count: {chunk_count}")

    # 3. Full Action Chunk (Title)
    texts_to_draw.append("3. Full Action Chunk:")
    
    # 3. Full Action Chunk (Content)
    if full_action_chunk is not None:
        for i, action_in_chunk in enumerate(full_action_chunk):
            # Highlight the current action being executed
            prefix = ">>" if i == current_action_index else "  "
            texts_to_draw.append(f"  {prefix} [{i}]: {np.round(action_in_chunk, 2)}")

    # 4. Position in Chunk
    position_str = f"4. Position in Chunk: {current_action_index + 1} / {len(full_action_chunk)}"
    texts_to_draw.append(position_str)

    # 5. Current Action
    texts_to_draw.append(f"5. Current Action: {np.round(current_action, 2)}")
    
    # 6. Next Action
    if next_action is not None:
        texts_to_draw.append(f"6. Next Action: {np.round(next_action, 2)}")
    else:
        texts_to_draw.append("6. Next Action: [End of Chunk]")
        
    # 7. Instruction
    texts_to_draw.append(f"7. Instruction: {instruction}")

    # --- Drawing Loop ---
    y0, dy_main, dy_chunk = 30, 30, 20
    y = y0
    
    for i, text in enumerate(texts_to_draw):
        is_chunk_content = text.strip().startswith('[') or text.strip().startswith('>>')
        
        if is_chunk_content:
            cv2.putText(frame_bgr, text, (10, y), chunk_font, chunk_font_scale, chunk_font_color, chunk_thickness, line_type)
            y += dy_chunk
        else:
            cv2.putText(frame_bgr, text, (10, y), main_font, main_font_scale, main_font_color, main_thickness, line_type)
            y += dy_main

    # Convert back to RGB for moviepy
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# ===============================================================================

def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"成功连接到 {args.remote_host}:{args.remote_port}!")

    df = pd.DataFrame(columns=["success", "duration", "is_follow", "object", "language_instruction", "video_filename"])

    while True:
        command_shortcuts = {
            "1": "pick up the corn on the plate",
            "2": "pick up the green pepper on the plate",
            "3": "pick up the red pepper on the plate",
            "4": "pick up the cabbage on the plate",
            "5": "pick up the potato on the plate",
            "6": "pick up the garlic on the plate",
        }
        raw_instruction = input("请输入语言指令 (1:corn, 2:green, 3:red, 4:cabbage, 5:potato, 6:garlic): ")
        instruction = command_shortcuts.get(raw_instruction, raw_instruction)
        print("当前执行的指令是:", instruction)
        
        executed_actions = []
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        chunk_count = 0

        third_video, wrist_left_video, wrist_right_video = [], [], []

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                s_time = time.time()
                curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=False)
                print(f"提取图片耗时 {time.time()- s_time} 秒")

                if t_step == 0:
                    images_to_save = [img for img in [image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224),
                                                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)] if img is not None]
                    if images_to_save:
                        combined_image = np.concatenate(images_to_save, axis=1)
                        Image.fromarray(combined_image).save("input_camera.png")
                
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224),
                        "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/state": np.concatenate((curr_obs["joint_position"], curr_obs["gripper_position"]), axis=0),
                        "prompt": instruction,
                    }
                    with prevent_keyboard_interrupt():
                        s_time = time.time()
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]
                        # print(f"推理一个chunk耗时 {time.time()- s_time} 秒")
                        chunk_count += 1

                action = pred_action_chunk[actions_from_chunk_completed]

                # ===============================================================================
                # MODIFIED VIDEO SAVING LOGIC
                # ===============================================================================
                current_action_index_in_chunk = actions_from_chunk_completed
                
                if current_action_index_in_chunk + 1 < len(pred_action_chunk):
                    next_action = pred_action_chunk[current_action_index_in_chunk + 1]
                else:
                    next_action = None

                # UPDATED: Call the modified helper function with all the new info
                if curr_obs["left_image"] is not None:
                    frame_with_info = add_info_to_frame(
                        frame=curr_obs["left_image"],
                        step=t_step,
                        chunk_count=chunk_count,
                        full_action_chunk=pred_action_chunk,
                        current_action_index=current_action_index_in_chunk,
                        current_action=action,
                        next_action=next_action,
                        instruction=instruction,
                    )
                    third_video.append(frame_with_info)

                if curr_obs["wrist_image"] is not None:
                    frame_with_info = add_info_to_frame(
                        frame=curr_obs["wrist_image"],
                        step=t_step,
                        chunk_count=chunk_count,
                        full_action_chunk=pred_action_chunk,
                        current_action_index=current_action_index_in_chunk,
                        current_action=action,
                        next_action=next_action,
                        instruction=instruction,
                    )
                    wrist_left_video.append(frame_with_info)
                # ===============================================================================

                actions_from_chunk_completed += 1
                
                action_to_execute = np.concatenate([action[:-1], np.ones((1,)) if action[-1].item() > 0.20 else np.zeros((1,))])
                env.step(action_to_execute)
                executed_actions.append(action_to_execute)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                break

        action_dir = "results/actions"
        if executed_actions:
            action_filename = os.path.join(action_dir, f"{datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}_actions.pkl")
            save_actions_to_pickle(executed_actions, action_filename)
        
        def save_video(frames, filename):
            if not frames: return
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(filename, codec="libx264", audio=False, logger=None)
            print(f"视频保存完成 {filename}")

        os.makedirs("results/video", exist_ok=True)
        print("正在保存视频....")
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video_stamp = timestamp
        save_video(third_video, os.path.join("results/video", f"{timestamp}_third_video.mp4"))
        save_video(wrist_left_video, os.path.join("results/video", f"{timestamp}_wrist_left_video.mp4"))
        save_video(wrist_right_video, os.path.join("results/video", f"{timestamp}_wrist_right_video.mp4"))

        success: Union[str, float, None] = None
        while not isinstance(success, float):
            success_input = input("这次测试成功了吗？(如果成功输入 y, 失败输入 n), 或是输入一个 0~1 的数值\n")
            if success_input.lower() == "y": success = 1.0
            elif success_input.lower() == "n": success = 0.0
            else:
                try:
                    success_val = float(success_input)
                    if 0.0 <= success_val <= 1.0: success = success_val
                    else: print("数字必须在 0 和 1 之间")
                except ValueError: print("无效输入，请输入 'y', 'n', 或一个 0 到 1 之间的数字。")
        
        import re
        KNOWN_OBJECTS = ["corn", "green pepper", "red pepper", "garlic", "potato", "cabbage"]
        def extract_object(instruction: str) -> Optional[str]:
            for obj in KNOWN_OBJECTS:
                if re.search(r"\b" + re.escape(obj.lower()) + r"\b", instruction.lower()): return obj
            return None
        
        is_follow = success == 1.0
        object_name = extract_object(instruction) if is_follow else input("请输入当前被错误抓取的物体名称: ")
        
        df = pd.concat(
            [df, pd.DataFrame([{"success": success, "duration": len(third_video), "is_follow": is_follow, "object": object_name, "language_instruction": instruction, "video_filename": f"{video_stamp}_video.mp4"}])],
            ignore_index=True
        )

        if input("再进行一次测试？(输入 y 或 n) ").lower() != "y": break
        env.reset()
        print("环境重设完成...")

    os.makedirs("results", exist_ok=True)
    csv_filename = os.path.join("results", f"eval_{datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"csv 文件保存完成，文件名为 {csv_filename}")
    print(df)


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, crop_ratios=CROP_RATIOS):
    from PIL import Image
    import numpy as np
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key, value in image_observations.items():
        if args.left_camera_id in key and "left" in key: left_image = value
        elif args.right_camera_id in key and "left" in key: right_image = value
        elif args.wrist_camera_id in key and "left" in key: wrist_image = value
    
    def process_image(img):
        if img is None: return None
        pil_img = Image.fromarray(img[..., :3][..., ::-1])
        if crop_ratios: pil_img = crop_left_right(pil_img, *crop_ratios)
        return np.array(pil_img)

    left_image, right_image, wrist_image = map(process_image, [left_image, right_image, wrist_image])

    robot_state = obs_dict["robot_state"]
    if save_to_disk:
        images_to_save = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if images_to_save: Image.fromarray(np.concatenate(images_to_save, axis=1)).save("robot_camera_views.png")

    return {
        "left_image": left_image, "right_image": right_image, "wrist_image": wrist_image,
        "cartesian_position": np.array(robot_state["cartesian_position"]),
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }

def save_actions_to_pickle(executed_actions, filename):
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'actions': np.array(executed_actions)}, f)
    print(f"✅ Actions successfully saved to {filename}")

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)