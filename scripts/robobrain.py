

import os
import time
import base64
import httpx
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
import tqdm


class RoboBrain:
    
    """
    可嵌入 RoboAgent 的视觉语言模块 (VLM)
    兼容接口：
        - set_logging(log_file, img_dir)
        - request_task(task_name, vl_inputs)
    """

    def __init__(self, port: int = 8000, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", api_key: str = "dummy"):

        transport = httpx.HTTPTransport(retries=1)
        self.client = OpenAI(
            base_url=f"http://162.105.195.74:{port}/v1",
            api_key=api_key,
            http_client=httpx.Client(transport=transport)
        )
        
        self.model = model_name
        self.logger = logging.getLogger("RoboBrain")
        self.logger.setLevel(logging.INFO)

        self.log_file = None
        self.img_dir = None

        print(f"✅ RoboBrain 初始化完成，连接端口: {port}, 模型: {model_name}")

    def set_logging(self, log_file, img_dir):
        """让 RoboAgent 的日志系统与 VLM 保持一致"""
        self.log_file = log_file
        self.img_dir = img_dir
        self.logger.info(f"VLM 日志绑定到: {getattr(log_file, 'name', 'None')}")

    def image_to_base64(self, image):
        """支持 numpy.ndarray 或 PIL.Image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_b64}"

    def request_task(self, task_name: str, vl_inputs: dict) -> str:
        """
        与 RoboAgent 对接的核心接口
        参数：
            task_name: str，例如 "prompt_completion_check"
            vl_inputs: dict，包含 images 与 user_prompt
        返回：
            模型生成的字符串
        """
        try:
            prompt = self._build_prompt(task_name, vl_inputs)
            messages = self._build_message(prompt, vl_inputs)

            t0 = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                timeout=60.0,
                stream=False
            )
            t1 = time.time()

            content = response.choices[0].message.content
            self.logger.info(f"[{task_name}] 推理耗时 {t1 - t0:.2f}s")
            self.logger.info(f"[{task_name}] 响应: {content[:100]}...")

            return content

        except Exception as e:
            self.logger.error(f"[{task_name}] 请求失败: {e}")
            return f"[ERROR] {e}"

    def _build_prompt(self, task_name, vl_inputs):

        user_prompt = vl_inputs.get("user_prompt", "")
        # return "描述这个图片"
        if task_name == "global_instruction_proposal":
            PROMPT = f"""You are an advanced robotic task planning expert with strong spatial reasoning, logical decomposition, and action sequence generation capabilities. 

You are controlling a robotic arm that needs to complete the following user prompt:{user_prompt}

**CONTEXT/KNOWLEDGE BASE CONSTRAINT:** **Only** the following six types of vegetables **can possibly exist** on the table: red pepper, green pepper, garlic, corn, cabbage, and potato. Your planning must be strictly limited to these six objects.

I will show you a image.Your task is to identify the relevant objects using your knowledge base	and analyze a complex robotic arm manipulation task and systematically break it down into a series of clear, executable sub-tasks.

Each item in the list must be a clear, natural language action instruction that describes a complete pick-and-place cycle for one object.

Return format:
- The output MUST be in English.
- The output must follow the structure below:

Example1:
User prompt: Put all the vegetables on the table into the plate.
You should refer to the following format for the output:
- Pick up the potato and place it on the plate.
- Pick up the cabbage and place it on the plate.
- Pick up the corn and place it on the plate.
- Pick up the garlic and place it on the plate.
- Pick up the green pepper and place it on the plate.
- Pick up the red pepper and place it on the plate.

Example2:
User prompt: Put all the peppers on the table into the plate.
You should refer to the following format for the output:
- Pick up the red pepper and place it on the plate.
- Pick up the green pepper and place it on the plate.

Now, it's your turn to output:
"""
            return PROMPT
        elif task_name == "dynamic_next_step_proposal":
            
            # 从 vl_inputs 中获取 "记忆"
            execution_history = vl_inputs.get("execution_history", []) 
            
            # 将列表格式化为 VLM 可读的字符串
            history_str = "\n".join([f"- {item}" for item in execution_history])
            if not history_str:
                history_str = "No steps taken yet."

            PROMPT = f"""You are an advanced robotic task planning agent with memory.
            
Your high-level goal is: '{user_prompt}'

You have a memory of previously executed steps and their outcomes:
{history_str}

Based on the *current image* and this *history*, what is the *single next* sub-task the robot should perform?

- If the history shows a *failure* (e.g., from a timeout or interference), you MUST decide how to recover. You can:
    1. **Retry** the *exact* same task.
    2. **Correct** the task (e.g., if "pick the can" failed, try "push the can").
    3. **Move on** to a different object (e.g., "pick up the apple").
- If the previous step was successful, decide the next logical step towards the goal.
- If you believe the high-level goal is *fully complete* by looking at the image, output the single word: DONE

Output only the single-line instruction for the *next* sub-task (e.g., "pick up the coke can") or "DONE".
"""
            return PROMPT
        
        elif task_name == "subtask_complete_check":
            PROMPT = f"""You are a meticulous robotic task evaluator. Your sole purpose is to verify if a given sub-task has been completed by observing the current state of the scene.

The robot was instructed to perform the following action: `{user_prompt}`

I will provide you with an image showing the scene *after* the robot attempted this action.

Your task is to analyze this image and determine if the goal state described in the sub-task is now true.

-   If the `current_sub_task` was "Pick up the potato and place it on the plate," you must check the image to confirm: Is the potato *on the plate*?
-   If the `current_sub_task` was "Pick up the green pepper and place it on the plate," you must check the image to confirm: Is the green pepper *on the new plate*?

You MUST return a single word only:
-   `True` if the task's goal state is achieved in the image.
-   `False` if the task's goal state is not achieved in the image."""
            return PROMPT
        elif task_name == "global_task_complete_check":
            PROMPT = f"""You are a meticulous final evaluator for a robotic task. Your sole purpose is to determine if the entire high-level goal has been successfully achieved by comparing the initial state to the final state.

You will be given three pieces of information:

1.  **Global Prompt:** The original user instruction.
    `{user_prompt}`

2.  **Initial Image:** An image showing the scene *before* the robot started any actions.

3.  **Current Image:** An image showing the scene *after* the robot supposedly finished all its actions.

Your task is to:

* Analyze the `{user_prompt}` and the `Initial Image` to understand the complete goal. (e.g., if the prompt is "put all vegetables on the plate," you must identify *all* vegetables that were on the table in the `Initial Image`).
* Verify that the `Current Image` shows the *final goal state* for *all* objects identified from the `Initial Image`, as required by the `{user_prompt}`.
* For example, if the goal is "put all vegetables on the plate," you must confirm that *every single vegetable* that was on the table in the `Initial Image` is now on the plate in the `Current Image`.

You MUST return a single word only:

* `True` if the global task's goal state is fully achieved in the `Current Image`.
* `False` if *any part* of the global task is incomplete (e.g., if the requirement is for all vegetables to be placed on the plate, but one vegetable has not yet been placed on the plate)."""

            return PROMPT
        else:
            raise ValueError(f"Unknown task_name: {task_name}")

    def _build_message(self, prompt, vl_inputs):
        """将文字 + 图像组装为多模态输入"""
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": prompt})

        if "images" in vl_inputs:
            for name, img in vl_inputs["images"].items():
                try:
                    img_b64_url = self.image_to_base64(img)
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": img_b64_url}})
                except Exception as e:
                    self.logger.warning(f"图像 {name} 转换失败: {e}")
        return messages


if __name__ == "__main__":

    test_img = "robot_camera_views_0.png"
    if not os.path.exists(test_img):
        print(f"❌ 找不到测试图像: {test_img}")
        exit()

    from PIL import Image
    img = np.array(Image.open(test_img))

    vlm = RoboBrain(port=8001)

    # # Initialize tqdm with the iterable (the range)
    # for i in tqdm.tqdm(range(10000)):
    #     out = vlm.request_task(
    #         task_name="global_instruction_proposal",
    #         vl_inputs={
    #             "images": {"current_image": img},
    #             "user_prompt": "Clear the items off the table and put them in the plate"
    #         }
    #     )

    #     print("\n🧠 模型输出 (最后一次迭代):")
    #     print(out)

    out = vlm.request_task(
        task_name="global_instruction_proposal",
        vl_inputs={
            "images": {"current_image": img},
            "user_prompt": "Clear the items off the table and put them in the plate"
        }
    )

    print("\n🧠 模型输出1:")
    print(out)

    out = vlm.request_task(
        task_name="subtask_complete_check",
        vl_inputs={
            "images": {"current_image": img},
            "user_prompt": "Pick up the corn and place it on the plate"
        }
    )
    print("\n🧠 模型输出2:")
    print(out)

    out = vlm.request_task(
        task_name="global_task_complete_check",
        vl_inputs={
            "images": {
                "initial_image": img,
                "current_image": img
            },
            "user_prompt": "Clear the items off the table and put them in the plate"
        }
    )
    print("\n🧠 模型输出3:")
    print(out)