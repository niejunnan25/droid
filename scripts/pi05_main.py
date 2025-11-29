# ruff: noqa

# pi05 推理最简版本
# 修改于 10 月 14 日 14:00

import dataclasses
import datetime
import faulthandler
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from droid.robot_env import RobotEnv
import tqdm
import tyro
from utils import prevent_keyboard_interrupt
from utils import TrajectoryCollector, save_video
faulthandler.enable()

DROID_CONTROL_FREQUENCY = 30

CROP_RATIOS = (0.27, 0.13)

COMMAND_SHORTCUTS = {
    "1": "pick up the corn on the plate",
    "2": "pick up the green pepper on the plate",
    "3": "pick up the red pepper on the plate",
    "4": "pick up the cabbage on the plate",
    "5": "pick up the potato on the plate",
    "6": "pick up the garlic on the plate",
}


@dataclasses.dataclass
class TimelineEvent:
    name: str
    kind: str
    start: datetime.datetime
    end: datetime.datetime
    extra: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start).total_seconds() * 1000.0


class EpisodeProfiler:
    """Records fine-grained timing statistics for each episode and writes a txt report."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = 0
        self.reset()

    def reset(self):
        self.events: list[TimelineEvent] = []
        self.global_start: Optional[datetime.datetime] = None
        self.global_end: Optional[datetime.datetime] = None
        self.seed: Optional[int] = None
        self.success: Optional[float] = None
        self.completed_steps: Optional[int] = None
        self.episode_slug: Optional[str] = None

    def start_episode(self, seed: int, *, episode_slug: Optional[str] = None):
        self.episode_idx += 1
        self.reset()
        self.seed = seed
        self.episode_slug = episode_slug
        self.global_start = datetime.datetime.now()

    def add_event(
        self,
        name: str,
        kind: str,
        start: datetime.datetime,
        end: datetime.datetime,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.events.append(TimelineEvent(name=name, kind=kind, start=start, end=end, extra=extra or {}))

    def end_episode(self, *, success: Optional[float] = None, completed_steps: Optional[int] = None):
        if self.global_start is None:
            return
        self.global_end = datetime.datetime.now()
        self.success = success
        self.completed_steps = completed_steps
        self._write_report()

    def _write_report(self):
        assert self.global_start is not None and self.global_end is not None
        timestamp_slug = self.episode_slug or self.global_start.strftime("%Y%m%d_%H%M%S")
        # Use timestamp-only naming; timestamp already uniquely identifies the episode and aligns with dataset slug
        log_path = self.log_dir / f"episode_{timestamp_slug}.txt"
        total_duration = (self.global_end - self.global_start).total_seconds()

        lines: list[str] = []
        lines.append(f"================ Episode {self.episode_idx:02d} (seed={self.seed}) ================")
        lines.append(f"Global Start : {self._format_datetime(self.global_start)}")
        lines.append(f"Global End   : {self._format_datetime(self.global_end)}")
        lines.append(f"Total Duration: {total_duration:.3f} s")
        lines.append("")
        lines.append("Timeline (strict order)")
        for idx, event in enumerate(self.events, 1):
            lines.append(self._format_event_line(idx, event))
        lines.append("")
        lines.append("Episode Summary")
        obs_count, obs_time = self._kind_stats("obs")
        pack_count, pack_time = self._kind_stats("pack")
        infer_count, infer_time = self._kind_stats("infer")
        action_count, action_time = self._kind_stats("action")
        others = max(total_duration - (obs_time + pack_time + infer_time + action_time), 0.0)
        lines.append(f"- obs 调用次数             : {obs_count} (总 {obs_time:.3f} s)")
        lines.append(f"- 数据打包次数             : {pack_count} (总 {pack_time:.3f} s)")
        lines.append(f"- 推理 chunk 数 / 耗时      : {infer_count} / {infer_time:.3f} s（client）")
        lines.append(f"- 执行动作次数 / 耗时       : {action_count} / {action_time:.3f} s")
        lines.append(f"- 其余时间（环境/等待）     : {others:.3f} s")
        if self.success is not None:
            lines.append(f"- success 标记             : {self.success}")
        if self.completed_steps is not None:
            lines.append(f"- 记录的 step 数           : {self.completed_steps}")
        lines.append("----------------------------------------------------------------")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[Profiler] Episode {self.episode_idx:02d} timeline saved to {log_path}")

    def _kind_stats(self, kind: str) -> tuple[int, float]:
        events = [e for e in self.events if e.kind == kind]
        total = sum(e.duration_ms for e in events) / 1000.0
        return len(events), total

    @staticmethod
    def _format_datetime(dt_obj: datetime.datetime) -> str:
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @staticmethod
    def _format_time(dt_obj: datetime.datetime) -> str:
        return dt_obj.strftime("%H:%M:%S.%f")[:-3]

    def _format_event_line(self, idx: int, event: TimelineEvent) -> str:
        start_str = self._format_time(event.start)
        end_str = self._format_time(event.end)
        duration = event.duration_ms
        gpu_ms = self._extract_gpu_latency(event.extra) if event.kind == "infer" else None
        # For infer events, show wall-clock/client time as Δ, GPU time as a separate column
        line = f"[{idx:3d}] {event.name:<35} {start_str} ~ {end_str}   Δ={duration:7.1f} ms"
        if event.kind == "infer" and gpu_ms is not None:
            line += f" | gpu={gpu_ms:6.1f} ms"
        return line

    @staticmethod
    def _extract_gpu_latency(extra: Optional[Dict[str, Any]]) -> Optional[float]:
        if not extra:
            return None
        for key in ("gpu_ms", "gpu_latency_ms", "gpu"):
            value = extra.get(key)
            if value is not None:
                return float(value)
        return None


def _extract_gpu_latency_from_response(response: Dict[str, Any]) -> Optional[float]:
    """Best-effort extraction of GPU latency metadata from policy server response."""
    # Prefer explicit policy_timing.infer_ms if present
    try:
        policy_timing = response.get("policy_timing", {}) if isinstance(response, dict) else {}
        if isinstance(policy_timing, dict) and "infer_ms" in policy_timing:
            return float(policy_timing["infer_ms"])
    except (TypeError, ValueError):
        pass
    candidate_keys = [
        ("metadata", "gpu_latency_ms"),
        ("metadata", "gpu_ms"),
        ("latency", "gpu_ms"),
        ("profiling", "gpu_ms"),
        ("profiling", "gpu_latency_ms"),
        (None, "gpu_latency_ms"),
        (None, "gpu_ms"),
    ]
    for parent, key in candidate_keys:
        if parent is None:
            container: Any = response
        else:
            container = response.get(parent) if isinstance(response, dict) else None
        if isinstance(container, dict) and key in container:
            try:
                return float(container[key])
            except (TypeError, ValueError):
                return None
    return None

@dataclasses.dataclass
class Args:
    external_camera: str = "left"

    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    max_timesteps: int = 1200
    open_loop_horizon: int = 18
    remote_host : str = "162.105.195.74"
    remote_port: int = 8000

    save_rollout: bool = False
    rollout_save_dir: str = "rollout/datasets"

    save_video: bool = False
    video_save_dir: str = "rollout/video"

    profile_logs: bool = True
    profile_log_dir: str = "rollout/episode_logs"

    write: bool = False  # 是否在保存的帧上写入 step/action/chunk 信息


def main(args: Args):

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")

    print("Created the droid env!")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"成功连接到 {args.remote_host}:{args.remote_port}!")

    collector = None
    profiler = None

    if args.save_rollout or args.save_video:
        collector = TrajectoryCollector(
            save_dir=args.rollout_save_dir,
            enable_rollout=args.save_rollout,
            enable_video=args.save_video,
            video_save_dir=args.video_save_dir,
            external_camera=args.external_camera,
            video_cameras=[f"{args.external_camera}_image", "wrist_image"],
            write=args.write,
        )

    if args.profile_logs:
        logs_dir = Path(args.profile_log_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        profiler = EpisodeProfiler(str(logs_dir))

    while True:
        episode_start_time = datetime.datetime.now()
        episode_slug = episode_start_time.strftime("%Y%m%d_%H%M%S")

        raw_instruction = input("请输入语言指令 (1:corn, 2:green, 3:red, 4:cabbage, 5:potato, 6:garlic): ")

        instruction = COMMAND_SHORTCUTS.get(raw_instruction, raw_instruction)

        print("当前执行的指令是:", instruction)

        if collector:
            collector.start_episode(instruction, episode_slug=episode_slug)

        episode_seed = int(np.random.randint(0, 1_000_000_000))
        if profiler:
            # Pass the same slug used by collector so log filenames align with saved datasets/videos
            profiler.start_episode(episode_seed, episode_slug=episode_slug)

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        current_chunk_id = 0
        current_chunk_size: int | None = None
        chunk_counter = 0
        obs_counter = 0
        action_counter = 0

        bar = tqdm.tqdm(range(args.max_timesteps))

        print("Running rollout... press Ctrl+C to stop early.")

        # Track how many steps were actually executed (useful for interrupted episodes)
        completed_steps = 0
        episode_success: Optional[float] = None

        try:
            for t_step in bar:
                start_time = time.time()

                try:
                    obs_counter += 1
                    obs_start = datetime.datetime.now()
                    raw_obs = env.get_observation()
                    curr_obs = _extract_observation(args, raw_obs, save_to_disk=False, crop_ratios=CROP_RATIOS)
                    obs_end = datetime.datetime.now()
                    if profiler:
                        profiler.add_event(f"obs#{obs_counter}", "obs", obs_start, obs_end)

                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        chunk_counter += 1

                        pack_start = datetime.datetime.now()
                        request_data = {
                            "observation/image": image_tools.resize_with_pad(
                                curr_obs[f"{args.external_camera}_image"], 224, 224
                            ),
                            "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                            "observation/state": np.concatenate(
                                (curr_obs["joint_position"], curr_obs["gripper_position"]), axis=0
                            ),
                            "prompt": instruction,
                        }
                        pack_end = datetime.datetime.now()
                        
                        if profiler:
                            profiler.add_event(f"pack+encode#{chunk_counter}", "pack", pack_start, pack_end)

                        infer_start = datetime.datetime.now()
                        with prevent_keyboard_interrupt():
                            infer_response = policy_client.infer(request_data)
                        infer_end = datetime.datetime.now()

                        pred_action_chunk = infer_response["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]
                        current_chunk_size = pred_action_chunk.shape[0]
                        current_chunk_id = chunk_counter

                        gpu_latency = _extract_gpu_latency_from_response(infer_response)
                        if profiler:
                            profiler.add_event(
                                f"infer chunk#{chunk_counter}",
                                "infer",
                                infer_start,
                                infer_end,
                                {"gpu_ms": gpu_latency} if gpu_latency is not None else {},
                            )

                    action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1
                    action_counter += 1

                    if action[-1].item() > 0.20:
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        action = np.concatenate([action[:-1], np.zeros((1,))])
                    s_t = time.time()
                    action_start = datetime.datetime.now()
                    env.step(action)
                    action_end = datetime.datetime.now()
                    print(f"动作执行耗时: {time.time() - s_t:.4f} 秒")

                    if profiler:
                        chunk_idx = actions_from_chunk_completed - 1
                        profiler.add_event(
                            f"action chunk#{current_chunk_id} idx{chunk_idx} (action#{action_counter})",
                            "action",
                            action_start,
                            action_end,
                        )

                    # print(f"第 {t_step} 步执行的动作: {action}")

                    completed_steps += 1

                    if collector:
                        collector.add_step(
                            curr_obs,
                            action,
                            step_idx=action_counter - 1,
                            chunk_idx=actions_from_chunk_completed,
                            chunk_size=current_chunk_size,
                        )

                    elapsed_time = time.time() - start_time
                    # print(f"Step time: {elapsed_time:.3f}s")

                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                except KeyboardInterrupt:
                    break
        finally:
            # Save trajectory/video when episode ends (either normally or via Ctrl+C)
            if collector:
                if collector.enable_rollout:
                    success = None
                    try:
                        while True:
                            user_in = input("这次测试成功吗？(y/n): ").strip().lower()
                            if user_in == "y":
                                success = 1.0
                                break
                            elif user_in == "n":
                                success = 0.0
                                break
                            else:
                                print("请输入 'y' 或 'n'。")
                    except KeyboardInterrupt:
                        print("\n标注中断，轨迹将标记为不完整（failure）。")
                        success = 0.0

                    collector.save_episode(success=success, completed_steps=completed_steps, episode_slug=episode_slug)
                    episode_success = success
                else:
                    # 视频-only 情况
                    collector.save_episode(success=None, completed_steps=completed_steps, episode_slug=episode_slug)

            if profiler:
                profiler.end_episode(success=episode_success, completed_steps=completed_steps)

            env.reset()
            print("环境重设完成...")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, crop_ratios=CROP_RATIOS):
    from PIL import Image
    import numpy as np

    # t0 = time.perf_counter()

    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # t_pick = time.perf_counter()
    
    # 这个地方拿到的是原始的相机 left_image, wrist_image

    def process_image(img):
        if img is None:
            return None
        img = img[..., :3][..., ::-1]  # 去 alpha & 转 RGB
        if crop_ratios:
            left_ratio, right_ratio = crop_ratios
            h, w = img.shape[:2]
            left = int(w * left_ratio)
            right = int(w * right_ratio)
            right_bound = max(left, w - right)
            img = img[:, left:right_bound]
        return img  # 已是 numpy RGB

    left_image = process_image(left_image)
    right_image = process_image(right_image)
    wrist_image = process_image(wrist_image)
    # t_process = time.perf_counter()
    # print(f"left_image shape: {left_image.shape}, wrist_image shape: {wrist_image.shape}")

    # 机器人状态
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])
    # t_state = time.perf_counter()

    if save_to_disk:
        images_to_save = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if images_to_save:
            combined_image = np.concatenate(images_to_save, axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

    # print(
    #     "[extract_obs] "
    #     f"pick_imgs={(t_pick - t0) * 1000:.2f}ms, "
    #     f"process_imgs={(t_process - t_pick) * 1000:.2f}ms, "
    #     f"state={(t_state - t_process) * 1000:.2f}ms, "
    #     f"total={(t_state - t0) * 1000:.2f}ms"
    # )

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
