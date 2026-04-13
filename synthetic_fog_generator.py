import os
import cv2
import numpy as np
from tqdm import tqdm


# ======================
# Depth Approximation
# ======================
def estimate_depth(img):
    # Use grayscale as a proxy for depth
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Invert to assume darker areas are farther
    depth = 1 - gray
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    return depth


# ======================
# Fog Generation (Koschmieder Model)
# 雾生成（基于 Koschmieder 大气散射模型）
# ======================
def add_fog_effect(img, beta):
    """
    I(x) = J(x)t(x) + A(1 - t(x))
    beta: Scattering coefficient (大气散射系数)
    """
    img = img.astype(np.float32) / 255.0
    depth = estimate_depth(img)

    # 1. Transmission map calculation (计算透射率函数 t)
    t = np.exp(-beta * depth)
    t = np.expand_dims(t, axis=2)

    # 2. Atmospheric light A (大气光项，随浓度略微提升亮度)
    A = 0.7 + 0.1 * beta
    A = np.clip(A, 0.7, 1.0)

    # 3. Apply scattering model (应用散射模型公式)
    fog_img = img * t + A * (1 - t)

    # 4. Optional: slight blur to simulate scattering (可选：轻微模糊模拟散射感)
    fog_img = cv2.GaussianBlur(fog_img, (3, 3), 0)

    fog_img = np.clip(fog_img, 0, 1)
    return (fog_img * 255).astype(np.uint8)


# ======================
# Batch Processing (批量处理)
# ======================
def process_fog(input_dir, output_base):
    # Scattering coefficients defined in the paper: beta
    # 论文中定义的大气散射系数: beta
    beta_settings = [0.5, 1.0, 1.5, 2.0]

    for beta in beta_settings:
        # Create output directory based on beta value (根据 beta 值创建输出目录)
        out_dir = os.path.join(output_base, f"fog_beta_{beta}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing Fog: beta={beta} ...")

        for img_name in tqdm(os.listdir(input_dir)):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Apply fog effect (应用雾化效果)
            fog_img = add_fog_effect(img, beta)

            save_path = os.path.join(out_dir, img_name)
            cv2.imwrite(save_path, fog_img)


# ======================
# Main Entry (主入口)
# ======================
if __name__ == "__main__":
    # Input folder path (输入文件夹路径)
    input_folder = r"E:\PoseCode\Cowdata_XuXS_test\val2017"

    # Output folder path (输出文件夹路径)
    output_folder = r"E:\PoseCode\Cowdata_XuXS_test\val2017_Fog_New"

    process_fog(input_folder, output_folder)