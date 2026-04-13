import os
import cv2
import numpy as np
from tqdm import tqdm


# ======================
# Rain Streak Simulation (雨丝模拟)
# ======================
def generate_rain_streak(img, N_count):
    h, w = img.shape[:2]
    rain_layer = np.zeros((h, w), dtype=np.float32)

    # Basic same direction for realistic rainfall (基本同方向，模拟真实降雨)
    angle = np.random.uniform(-np.pi / 18, np.pi / 18)

    for _ in range(N_count):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        # Length and thickness slightly adjusted by N to reflect intensity
        # 长度和厚度随N略微调整以反映强度感
        length = np.random.randint(20, 50)
        thickness = 1 if N_count < 500 else 2

        dx = int(length * np.sin(angle))
        dy = int(length * np.cos(angle))

        x2 = np.clip(x + dx, 0, w - 1)
        y2 = np.clip(y + dy, 0, h - 1)

        cv2.line(rain_layer, (x, y), (x2, y2), 255, thickness)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    return rain_layer / 255.0


# ======================
# Core Function: Rainfall Synthesis
# 主函数：降雨合成
# ======================
def add_rain_effect(img, alpha, N_count, eta):
    img = img.astype(np.float32) / 255.0

    # 1. Brightness Attenuation (alpha) - 亮度衰减
    img_attenuated = img * alpha

    # 2. Rain Streaks (N) - 雨丝噪声
    rain_streak = generate_rain_streak(img, N_count)
    rain_streak = np.expand_dims(rain_streak, axis=2)
    # Adding rain streaks with brightness influence (增强雨丝视觉效果)
    img_rain = img_attenuated + rain_streak * 0.5

    # 3. Fogging effect (eta) - 雾化项
    fog = np.ones_like(img) * eta
    img_final = img_rain * (1 - fog) + fog

    img_final = np.clip(img_final, 0, 1)
    return (img_final * 255).astype(np.uint8)


# ======================
# Batch Processing (批量处理)
# ======================
def process_rain(input_dir, output_base):
    # Joint control settings from the paper: (alpha, N, eta)
    # 论文中的联合控制设置：(亮度衰减, 雨丝数量, 雾化系数)
    rain_settings = [
        (0.92, 134, 0.04),  # Level 1
        (0.84, 379, 0.08),  # Level 2
        (0.76, 697, 0.12),  # Level 3
        (0.68, 1073, 0.16)  # Level 4
    ]

    for alpha, N, eta in rain_settings:
        out_dir = os.path.join(output_base, f"rain_a{alpha}_N{N}_e{eta}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing Rain: alpha={alpha}, N={N}, eta={eta}")
        for img_name in tqdm(os.listdir(input_dir)):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(input_dir, img_name))
                if img is not None:
                    res = add_rain_effect(img, alpha, N, eta)
                    cv2.imwrite(os.path.join(out_dir, img_name), res)


if __name__ == "__main__":
    input_path = r"E:\PoseCode\Cowdata_XuXS_test\val2017"
    output_path = r"E:\PoseCode\Cowdata_XuXS_test\val2017_Rain_New"
    process_rain(input_path, output_path)