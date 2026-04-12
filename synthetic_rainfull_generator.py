import os
import cv2
import numpy as np
from tqdm import tqdm

def lowlight_transform(img, gamma, alpha):
    img = img.astype(np.float32) / 255.0

    # 1. Gamma Transformation (gamma) - Gamma变换
    img_gamma = np.power(img, gamma)

    # 2. Global Brightness Attenuation (alpha) - 全局亮度衰减
    img_dark = img_gamma * alpha

    # 3. Add slight imaging noise (模拟低光成像噪声)
    noise = np.random.normal(0, 0.01, img_dark.shape)
    img_final = np.clip(img_dark + noise, 0, 1)

    return (img_final * 255).astype(np.uint8)

def process_lowlight(input_dir, output_base):
    # Settings from the paper: (gamma, alpha)
    # 论文中的参数设置：(Gamma参数, 亮度系数)
    settings = [
        (1.2, 0.9), # Level 1
        (1.5, 0.7), # Level 2
        (2.0, 0.5), # Level 3
        (2.5, 0.4)  # Level 4
    ]

    for gamma, alpha in settings:
        out_dir = os.path.join(output_base, f"lowlight_g{gamma}_a{alpha}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing Low-light: gamma={gamma}, alpha={alpha}")
        for img_name in tqdm(os.listdir(input_dir)):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(input_dir, img_name))
                if img is not None:
                    res = lowlight_transform(img, gamma, alpha)
                    # Apply slight blur to simulate sensor characteristics (轻微模糊模拟传感器特性)
                    res = cv2.GaussianBlur(res, (3, 3), 0)
                    cv2.imwrite(os.path.join(out_dir, img_name), res)

if __name__ == "__main__":
    input_path = r"E:\PoseCode\Cowdata_XuXS_test\val2017"
    output_path = r"E:\PoseCode\Cowdata_XuXS_test\val2017_LowLight"
    process_lowlight(input_path, output_path)