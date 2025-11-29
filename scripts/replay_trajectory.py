#!/usr/bin/env python3
"""
Replay saved actions from a pickle file.

Supports:
- Dict with key "actions" (e.g., trajectory.pkl from TrajectoryCollector or actions.pkl)
- Direct list/ndarray of actions

Usage examples:
  python scripts/replay_trajectory.py rollout/datasets/episode_20251128_142342/trajectory.pkl
  python scripts/replay_trajectory.py results/actions/2025_10_15_23:22:55_actions.pkl --freq 4 --max-steps 500
"""

import argparse
import faulthandler
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from droid.robot_env import RobotEnv

faulthandler.enable()


def load_actions(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "actions" in data:
        actions = data["actions"]
    elif isinstance(data, (list, tuple, np.ndarray)):
        actions = data
    else:
        raise ValueError("文件内容既不是包含 'actions' 键的字典，也不是动作数组/列表。")

    actions_arr = np.array(actions)
    if actions_arr.ndim != 2:
        raise ValueError(f"动作数组期望二维 (T, DoF)，但得到 shape={actions_arr.shape}")
    return actions_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay actions from a pickle file.")
    parser.add_argument("pickle_path", type=Path, help="路径到包含动作的 pickle 文件（trajectory.pkl 或 actions.pkl）。")
    parser.add_argument("--freq", type=float, default=15, help="控制频率 Hz（默认 15Hz）。")
    parser.add_argument("--max-steps", type=int, default=None, help="最多回放多少步；默认回放全部。")
    parser.add_argument("--start", type=int, default=0, help="从第几步开始回放（默认 0）。")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        env = RobotEnv(action_space="joint_position", gripper_action_space="position")
        print("✅ 机器人环境初始化成功！")
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    try:
        actions = load_actions(args.pickle_path)
    except Exception as e:
        print(f"❌ 加载动作失败: {e}")
        return

    if args.start >= len(actions):
        print(f"❌ start={args.start} 超过动作序列长度 {len(actions)}")
        return

    available = len(actions) - args.start
    total_steps = available if args.max_steps is None else min(args.max_steps, available)
    print(f"✅ 成功加载动作: shape={actions.shape}，将回放 {total_steps} 步（从 {args.start} 开始）@ {args.freq} Hz")

    try:
        for i in tqdm.tqdm(range(total_steps), desc="正在执行动作"):
            start_time = time.time()

            action = actions[args.start + i]
            env.step(action)
            print(f"当前 {i + args.start} 步, 执行的动作是 {action}")

            elapsed_time = time.time() - start_time
            sleep_duration = (1 / args.freq) - elapsed_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n🛑 用户通过 Ctrl+C 手动停止了程序。")
    except Exception as e:
        print(f"\n❌ 控制循环中发生错误: {e}")
    finally:
        print("正在重置机器人环境...")
        try:
            env.reset()
            print("✅ 程序执行完毕，环境已重置。")
        except Exception as e:
            print(f"⚠️ 重置时出错: {e}")


if __name__ == "__main__":
    main()
