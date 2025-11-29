import copy
import os

import cv2
import numpy as np
import pyzed.sl as sl
from PIL import Image

from utils import crop_left_right

class ZedCamera:
    def __init__(self, serial_number=None, device_path=None, resolution=sl.RESOLUTION.HD1080, fps=30):
        """
        初始化 ZED 相机，设置分辨率和帧率。

        参数：
            resolution: ZED 相机分辨率（默认：HD1080）
            fps: 帧率（默认：30）
        """

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = 30

        # 设置设备输入来源（按优先级选择）
        if serial_number is not None:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init_params.input = input_type
        elif device_path is not None:
            init_params.input = sl.InputType(device_path)

        # 打开相机
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"无法打开 ZED 相机：{err}")
            exit()

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

        self.frame_count = 0

    def capture_frame(self):
        """
        捕获左右目相机的一帧图像。

        返回：
            tuple: (left_image, right_image) 左右目图像的 NumPy 数组，若捕获失败返回 (None, None)
        """
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 获取左右目图像
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

            # 转换为 NumPy 数组, 去掉 alpha 通道
            left_image = self.image_left.get_data()[:,:,:3]
            right_image = self.image_right.get_data()[:,:,:3]

            self.frame_count += 1

            new_left_image = copy.deepcopy(left_image)
            new_right_image = copy.deepcopy(right_image)

            return new_left_image, new_right_image
        else:
            print("捕获帧失败")
            return None, None

    def close(self):
        """
        关闭 ZED 相机并释放资源。
        """
        self.zed.close()
        print("ZED Camera has been closed!")

    def __del__(self):

        self.close()

def show_live_preview(camera: ZedCamera, window_name: str = "ZED Preview"):
    """
    使用 OpenCV 窗口实时显示单个 ZED 相机的裁剪画面。
    """
    LEFT_RATIO = 0.27
    RIGHT_RATIO = 0.13

    try:
        while True:
            left_img, _ = camera.capture_frame()
            if left_img is None:
                continue
            # 转换到 PIL 做裁剪，再转回 BGR 以方便 OpenCV 显示
            left_pil = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
            cropped_pil = crop_left_right(left_pil, LEFT_RATIO, RIGHT_RATIO)
            cropped_rgb = np.array(cropped_pil)
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, cropped_bgr)

            # 按下 q 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        camera.close()


if __name__ == "__main__":
    # 只预览腕部相机 36276705
    # zed_wrist_camera = ZedCamera(serial_number=13132609)
    zed_camera = ZedCamera(serial_number=36276705)
    show_live_preview(zed_camera, window_name="Wrist Camera")
