import os
import cv2
import numpy as np
from tqdm import tqdm


# ======================
# Low-light Simulation (Gamma + Brightness Attenuation)
# 低光模拟（Gamma + 亮度衰减）
# ======================
def lowlight_transform(img, gamma, alpha):
    img = img.astype(np.float32) / 255.0

    # 1️⃣ Gamma Transformation (Gamma变换)
    img_gamma = np.power(img, gamma)

    # 2️⃣ Global Brightness Attenuation (全局亮度衰减)
    img_dark = img_gamma * alpha

    # 3️⃣ Optional: Add slight noise for more realistic low-light (可选：加入轻微噪声使更真实)
    noise = np.random.normal(0, 0.02, img_dark.shape)
    img_dark = img_dark + noise

    img_dark = np.clip(img_dark, 0, 1)
    img_dark = (img_dark * 255).astype(np.uint8)

    return img_dark


# ======================
# Batch Processing (批量处理)
# ======================
def process_folder(input_dir, output_base):
    # 👉 Four low-light levels consistent with the paper (四个低光等级，与论文一致)
    settings = [
        (1.2, 0.9),  # L1
        (1.5, 0.7),  # L2
        (2.0, 0.5),  # L3
        (2.5, 0.4)  # L4
    ]

    for gamma, alpha in settings:
        out_dir = os.path.join(output_base, f"lowlight_g{gamma}_a{alpha}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing gamma={gamma}, alpha={alpha} ...")

        for img_name in tqdm(os.listdir(input_dir)):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            out_img = lowlight_transform(img, gamma, alpha)

            # 👉 Optional: Slight blur to simulate low-light imaging (可选：轻微模糊模拟低光成像)
            out_img = cv2.GaussianBlur(out_img, (3, 3), 0)

            save_path = os.path.join(out_dir, img_name)
            cv2.imwrite(save_path, out_img)


# ======================
# Main Entry (主入口)
# ======================
if __name__ == "__main__":
    # Original image path (原始图像路径)
    input_folder = r"E:\PoseCode\Cowdata_XuXS_test\val2017"

    # Output path (输出路径)
    output_folder = r"E:\PoseCode\Cowdata_XuXS_test\val2017_LowLight"

    process_folder(input_folder, output_folder)