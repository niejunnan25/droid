# ruff: noqa

# pi05 推理极致优化版本 - 只在需要时提取图片

# 修改于 11 月 27 日

import contextlib
import dataclasses
import faulthandler
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional
faulthandler.enable()

from pi05_main import (
    _extract_observation,
    COMMAND_SHORTCUTS
)
from utils import prevent_keyboard_interrupt

DROID_CONTROL_FREQUENCY = 50

CROP_RATIOS = (0.27, 0.13)

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    # Policy parameters
    external_camera: Optional[str] = "left"

    # Rollout parameters
    max_timesteps: int = 1200
    open_loop_horizon: int = 18

    # Remote server parameters
    remote_host: str = "162.105.195.74"
    remote_port: int = 8000

def main(args: Args):

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("✓ Created the droid env!")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"✓ Connected to {args.remote_host}:{args.remote_port}!")

    while True:

        raw_instruction = input("请输入语言指令 (1:corn, 2:green, 3:red, 4:cabbage, 5:potato, 6:garlic): ")

        instruction = COMMAND_SHORTCUTS.get(raw_instruction, raw_instruction)
        print(f"执行指令: {instruction}\n")

        actions_from_chunk_completed = 0
        pred_action_chunk = None

        bar = tqdm.tqdm(range(args.max_timesteps), desc="执行中")

        try:
            for t_step in bar:
                start_time = time.time()

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    s_time = time.time()
                    curr_obs = _extract_observation(args, env.get_observation())
                    extract_time = time.time() - s_time

                    # 构建请求
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224),
                        "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/state": np.concatenate((curr_obs["joint_position"], curr_obs["gripper_position"]), axis=0),
                        "prompt": instruction,
                    }

                    # 推理
                    with prevent_keyboard_interrupt():
                        s_time = time.time()
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]
                        infer_time = time.time() - s_time

                    bar.set_postfix({"提取": f"{extract_time:.3f}s", "推理": f"{infer_time:.3f}s"})

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if action[-1].item() > 0.20:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        except KeyboardInterrupt:
            print("\n⏸ 任务中断")

        finally:
            env.reset()
            print("✓ 环境重设完成\n")

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
