# ruff: noqa

import dataclasses
import datetime
import faulthandler
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from droid.robot_env import RobotEnv
from utils import TrajectoryCollector, prevent_keyboard_interrupt

from pi05_main import (
    COMMAND_SHORTCUTS,
    EpisodeProfiler,
    CROP_RATIOS,
    DROID_CONTROL_FREQUENCY,
    _extract_gpu_latency_from_response,
    _extract_observation,
)

from ZEDCamera import ZedCamera

faulthandler.enable()


@dataclasses.dataclass
class BaseArgs:
    external_camera: str = "left"

    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    max_timesteps: int = 1200
    open_loop_horizon: int = 18
    remote_host: str = "162.105.195.74"
    remote_port: int = 8000

    save_rollout: bool = False
    rollout_save_dir: str = "rollout/datasets"

    save_video: bool = False
    video_save_dir: str = "rollout/video"

    profile_logs: bool = True
    profile_log_dir: str = "rollout/episode_logs"

    use_env_camera: bool = False # 是否由 RobotEnv 初始化/读取相机；False 时使用 ZedCamera 手动抓取

    write: bool = False  # 是否在保存的帧上写入 step/action/chunk 信息


class ManualCameraManager:

    """Lightweight ZED 相机读取器，可在不依赖 RobotEnv 相机的情况下抓图。"""

    def __init__(self, args: BaseArgs):
        self.args = args
        self.external_serial = self._resolve_external_serial(args)
        self.wrist_serial = args.wrist_camera_id

        self.external_camera = self._init_camera(self.external_serial, "external") if self.external_serial else None
        self.wrist_camera = self._init_camera(self.wrist_serial, "wrist")

        if self.wrist_camera is None:
            raise RuntimeError("未能初始化腕部相机（wrist_camera_id 无效或初始化失败）")

    def _resolve_external_serial(self, args: BaseArgs) -> Optional[str]:
        if args.external_camera == "right":
            return args.right_camera_id
        return args.left_camera_id

    def _init_camera(self, serial: str | None, label: str) -> Optional[ZedCamera]:
        if not serial or serial == "<your_camera_id>":
            return None
        try:
            return ZedCamera(serial_number=int(serial))
        except Exception as exc:
            raise RuntimeError(f"初始化 {label} ZED 相机失败（serial={serial}）: {exc}") from exc

    def build_observation(self, env: RobotEnv) -> Dict[str, Any]:
        state_dict, _ = env.get_state()
        images: Dict[str, np.ndarray] = {}

        if self.external_camera:
            ext_left, _ = self.external_camera.capture_frame()
            if ext_left is not None:
                images[f"{self.external_serial}_left"] = ext_left

        if self.wrist_camera:
            wrist_left, _ = self.wrist_camera.capture_frame()
            if wrist_left is not None:
                images[f"{self.wrist_serial}_left"] = wrist_left

        return {"robot_state": state_dict, "image": images}

    def close(self):
        try:
            if self.external_camera:
                self.external_camera.close()
        finally:
            if self.wrist_camera:
                self.wrist_camera.close()


class AsyncObsFetcher:
    """Background fetcher: continuously pulls camera images (no robot state), keeps only the latest frame."""

    def __init__(self, env: RobotEnv, args: "Args", manual_cam_mgr: Optional[ManualCameraManager] = None):
        self.env = env
        self.args = args
        self.manual_cam_mgr = manual_cam_mgr

        self._latest: Optional[Tuple[Dict[str, Any], float]] = None  # {"image": {...}}, timestamp
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.interrupted = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AsyncObsFetcher", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[Tuple[Dict[str, Any], float]]:
        with self._lock:
            return self._latest

    def clear_latest(self):
        with self._lock:
            self._latest = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _fetch_once(self) -> Dict[str, Any]:
        """Only fetch images to avoid cross-thread RPC on robot state."""
        if self.args.use_env_camera:
            camera_obs, _ = self.env.read_cameras()
            obs = {}
            obs.update(camera_obs)
            obs.setdefault("image", {})
            return obs

        assert self.manual_cam_mgr is not None
        images: Dict[str, Any] = {}
        if self.manual_cam_mgr.external_camera:
            ext_left, _ = self.manual_cam_mgr.external_camera.capture_frame()
            if ext_left is not None:
                images[f"{self.manual_cam_mgr.external_serial}_left"] = ext_left
        if self.manual_cam_mgr.wrist_camera:
            wrist_left, _ = self.manual_cam_mgr.wrist_camera.capture_frame()
            if wrist_left is not None:
                images[f"{self.manual_cam_mgr.wrist_serial}_left"] = wrist_left
        return {"image": images}

    def _run(self):
        target_interval = 1.0 / self.args.fetch_hz if self.args.fetch_hz > 0 else 0.0
        while not self._stop_event.is_set():
            start = time.time()
            try:
                obs = self._fetch_once()
                with self._lock:
                    self._latest = (obs, time.time())
            except KeyboardInterrupt:
                self.interrupted = True
                self._stop_event.set()
                break
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
                self._stop_event.set()
                break

            if target_interval > 0:
                elapsed = time.time() - start
                sleep_for = target_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)


@dataclasses.dataclass
class Args(BaseArgs):
    fetch_hz: float = 30.0
    stale_frame_warn_ms: float = 300.0
    first_frame_timeout_s: float = 5.0
    self_test: bool = False


def main(args: Args):
    if args.self_test:
        run_self_test(args)
        return

    if not args.use_env_camera:
        from droid.camera_utils.wrappers import multi_camera_wrapper as mcw

        mcw.gather_zed_cameras = lambda: []
        print("已禁用 RobotEnv 内置相机初始化，将使用 ZEDCamera 手动抓取。")

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    manual_cam_mgr = None
    if not args.use_env_camera:
        manual_cam_mgr = ManualCameraManager(args)

    fetcher = AsyncObsFetcher(env, args, manual_cam_mgr)
    fetcher.start()

    first_wait_start = time.time()
    while fetcher.get_latest() is None:
        if fetcher.error:
            raise RuntimeError(f"后台抓取异常：{fetcher.error}")  # surface failure early
        if fetcher.interrupted:
            raise KeyboardInterrupt
        if args.first_frame_timeout_s and (time.time() - first_wait_start) > args.first_frame_timeout_s:
            raise TimeoutError(f"{args.first_frame_timeout_s}s 内未抓到首帧，请检查相机/机器人连接")
        print("等待首帧抓取...")
        time.sleep(0.2)

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

    try:
        while True:

            if not fetcher.is_running():
                fetcher.stop()
                fetcher = AsyncObsFetcher(env, args, manual_cam_mgr)
                fetcher.start()
            fetcher.clear_latest()

            episode_start_time = datetime.datetime.now()
            episode_slug = episode_start_time.strftime("%Y%m%d_%H%M%S")

            raw_instruction = input("请输入语言指令 (1:corn, 2:green, 3:red, 4:cabbage, 5:potato, 6:garlic): ")

            instruction = COMMAND_SHORTCUTS.get(raw_instruction, raw_instruction)

            print("当前执行的指令是:", instruction)

            if collector:
                collector.start_episode(instruction, episode_slug=episode_slug)

            episode_seed = int(np.random.randint(0, 1_000_000_000))
            if profiler:
                profiler.start_episode(episode_seed, episode_slug=episode_slug)

            actions_from_chunk_completed = 0
            pred_action_chunk = None
            current_chunk_id = 0
            current_chunk_size: int | None = None
            chunk_counter = 0
            obs_counter = 0
            action_counter = 0

            bar = tqdm.tqdm(total=args.max_timesteps)

            print("Running rollout... press Ctrl+C to stop early.")

            completed_steps = 0
            episode_success: Optional[float] = None

            try:
                while True:
                    if fetcher.error:
                        raise RuntimeError(f"后台抓取异常: {fetcher.error}")
                    if fetcher.interrupted:
                        raise KeyboardInterrupt
                    
                    # s_t = time.time()
                    latest = fetcher.get_latest()
                    # print(f"抓取最新帧耗时: {time.time() - s_t:.4f} 秒")

                    if latest is None:
                        time.sleep(0.001)
                        continue

                    image_obs, obs_ts = latest

                    # 状态放在主线程同步获取，避免跨线程 RPC
                    # s_t = time.time()
                    state_dict, _ = env.get_state()
                    # print(f"状态获取耗时: {time.time() - s_t:.4f} 秒")

                    raw_obs = {"robot_state": state_dict}
                    raw_obs.update(image_obs)
                    raw_obs.setdefault("image", {})
                    obs_counter += 1

                    obs_start = datetime.datetime.now()
                    curr_obs = _extract_observation(args, raw_obs, save_to_disk=False, crop_ratios=CROP_RATIOS)
                    obs_end = datetime.datetime.now()

                    if profiler:
                        profiler.add_event(f"obs#{obs_counter}", "obs", obs_start, obs_end, {"age_ms": (time.time() - obs_ts) * 1000.0},)

                    age_ms = (time.time() - obs_ts) * 1000.0
                    if args.stale_frame_warn_ms and age_ms > args.stale_frame_warn_ms:
                        print(f"⚠️ 当前帧滞后 {age_ms:.1f} ms")

                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        chunk_counter += 1

                        pack_start = datetime.datetime.now()
                        request_data = {
                            "observation/image": image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224),
                            "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                            "observation/state": np.concatenate((curr_obs["joint_position"], curr_obs["gripper_position"]), axis=0),
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
                            profiler.add_event(f"infer chunk#{chunk_counter}", "infer", infer_start, infer_end, {"gpu_ms": gpu_latency} if gpu_latency is not None else {})

                    action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1
                    action_counter += 1

                    if action[-1].item() > 0.20:
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        action = np.concatenate([action[:-1], np.zeros((1,))])
                    
                    action_start = datetime.datetime.now()
                    env.step(action)
                    action_end = datetime.datetime.now()

                    if profiler:
                        chunk_idx = actions_from_chunk_completed - 1
                        profiler.add_event(
                            f"action chunk#{current_chunk_id} idx{chunk_idx} (action#{action_counter})",
                            "action",
                            action_start,
                            action_end,
                        )

                    completed_steps += 1

                    if collector:
                        collector.add_step(
                            curr_obs,
                            action,
                            step_idx=action_counter - 1,
                            chunk_idx=actions_from_chunk_completed,
                            chunk_size=current_chunk_size,
                        )

                    elapsed_time = time.time() - obs_ts
                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(max(0.0, 1 / DROID_CONTROL_FREQUENCY - elapsed_time))

                    if completed_steps >= args.max_timesteps:
                        break
                    bar.update(1)
            except KeyboardInterrupt:
                print("\n🛑 当前 episode 手动中断")
            finally:
                bar.close()
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
                        collector.save_episode(success=None, completed_steps=completed_steps, episode_slug=episode_slug)

                if profiler:
                    profiler.end_episode(success=episode_success, completed_steps=completed_steps)

                env.reset()
                print("环境重设完成...")
    except KeyboardInterrupt:
        print("\n🛑 收到退出请求，正在清理资源...")
    finally:
        fetcher.stop()
        if manual_cam_mgr:
            manual_cam_mgr.close()


def run_self_test(args: Args):
    """单次自检：分别测试状态获取与相机抓取，快速判断卡点。"""
    print("=== 自检模式：检查状态获取与相机抓取 ===")

    if not args.use_env_camera:
        from droid.camera_utils.wrappers import multi_camera_wrapper as mcw

        mcw.gather_zed_cameras = lambda: []
        print("已禁用 RobotEnv 内置相机初始化，将使用 ZEDCamera 手动抓取。")

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    manual_cam_mgr = None
    if not args.use_env_camera:
        manual_cam_mgr = ManualCameraManager(args)

    def try_state():
        print("\n[自检] 调用 env.get_state() ...")
        t0 = time.time()
        try:
            state, _ = env.get_state()
            dt = time.time() - t0
            print(f"[自检] get_state 成功，用时 {dt:.3f}s，键：{list(state.keys())}")
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[自检] get_state 失败，用时 {dt:.3f}s，异常: {exc}")

    def try_env_cameras():
        print("\n[自检] 调用 env.read_cameras() ...")
        t0 = time.time()
        try:
            cam_obs, _ = env.read_cameras()
            dt = time.time() - t0
            images = cam_obs.get("image", {})
            print(f"[自检] env.read_cameras 成功，用时 {dt:.3f}s，可用图像键: {list(images.keys())}")
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[自检] env.read_cameras 失败，用时 {dt:.3f}s，异常: {exc}")

    def try_manual_cameras():
        print("\n[自检] 调用 ManualCameraManager 手动抓取 ...")
        t0 = time.time()
        try:
            images: Dict[str, Any] = {}
            if manual_cam_mgr and manual_cam_mgr.external_camera:
                ext_left, _ = manual_cam_mgr.external_camera.capture_frame()
                if ext_left is not None:
                    images[f"{manual_cam_mgr.external_serial}_left"] = ext_left
            if manual_cam_mgr and manual_cam_mgr.wrist_camera:
                wrist_left, _ = manual_cam_mgr.wrist_camera.capture_frame()
                if wrist_left is not None:
                    images[f"{manual_cam_mgr.wrist_serial}_left"] = wrist_left
            dt = time.time() - t0
            shapes = {k: v.shape for k, v in images.items()}
            print(f"[自检] 手动抓取完成，用时 {dt:.3f}s，图像键: {list(images.keys())}, 形状: {shapes}")
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[自检] 手动抓取失败，用时 {dt:.3f}s，异常: {exc}")

    try_state()
    if args.use_env_camera:
        try_env_cameras()
    else:
        try_manual_cameras()

    if manual_cam_mgr:
        manual_cam_mgr.close()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
