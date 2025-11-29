import contextlib
import dataclasses
import datetime
import faulthandler
import numpy as np
import pandas as pd
import signal
import time
import tqdm
import tyro
from typing import Optional
from droid.robot_env import RobotEnv

faulthandler.enable()

# DROID control frequency
DROID_CONTROL_FREQUENCY = 15

@dataclasses.dataclass
class Args:
    # Rollout parameters
    max_timesteps: int = 100  # Reduced since we're doing single-step motion
    target_pose: str = "0.5,0.0,0.5,0.0,3.14,0.0"  # x,y,z,rx,ry,rz
    gripper_state: float = 0.0  # 0.0 for open, 1.0 for closed

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts."""
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

def main(args: Args):
    # Initialize the Panda environment with cartesian position action space
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    print("Created the droid env with cartesian position control!")

    # Parse target pose
    try:
        target_pose = np.array([float(x) for x in args.target_pose.split(",")])
        if len(target_pose) != 6:
            raise ValueError("Target pose must have 6 values (x,y,z,rx,ry,rz)")
    except ValueError as e:
        print(f"Error parsing target pose: {e}")
        return

    df = pd.DataFrame(columns=["success", "duration", "final_pose_error"])

    while True:
        # Reset environment
        env.reset()
        
        # Prepare action
        action = np.concatenate([target_pose, [args.gripper_state]])
        action = np.clip(action, -1, 1)  # Clip action to valid range

        # Rollout
        t_step = 0
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Executing motion... press Ctrl+C to stop early.")
        start_time = time.time()
        
        try:
            with prevent_keyboard_interrupt():
                for t_step in bar:
                    # Get current observation
                    obs = env.get_observation()
                    curr_pose = np.array(obs["robot_state"]["cartesian_position"])
                    
                    # Execute action
                    env.step(action)
                    
                    # Check if we're close enough to target
                    pose_error = np.linalg.norm(curr_pose - target_pose)
                    if pose_error < 0.01:  # 1cm tolerance
                        print("Reached target pose!")
                        break

                    # Sleep to match DROID control frequency
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                    
                    start_time = time.time()

        except KeyboardInterrupt:
            print("Rollout interrupted by user")
            break

        # Evaluate success
        final_obs = env.get_observation()
        final_pose = np.array(final_obs["robot_state"]["cartesian_position"])
        final_error = np.linalg.norm(final_pose - target_pose)
        success = 1.0 if final_error < 0.01 else 0.0

        # Log results
        df = pd.concat([df, pd.DataFrame({
            "success": [success],
            "duration": [t_step],
            "final_pose_error": [final_error]
        })], ignore_index=True)

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)