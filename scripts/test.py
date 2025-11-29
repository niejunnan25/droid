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

env = RobotEnv(action_space="joint_position", gripper_action_space="position")

print(env.get_observation())