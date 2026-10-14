
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)

# ==========================================
# 1. Load Depth Anything V2
# ==========================================

model_name = "depth-anything/Depth-Anything-V2-Small-hf"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Device:", device)

# ==========================================
# 2. Read image
# ==========================================

image = Image.open("1791884505248_2158353805548525754_953656442139000946_234c3be9e209db03efa0b5a9cbb33cfe.jpg").convert("RGB")

original_width, original_height = image.size

print("Original image size:", image.size)

# ==========================================
# 3. Predict depth
# ==========================================

inputs = processor(images=image, return_tensors="pt")
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

# Resize depth map về kích thước ảnh gốc
predicted_depth = F.interpolate(
    outputs.predicted_depth.unsqueeze(1),
    size=(original_height, original_width),
    mode="bilinear",
    align_corners=False
).squeeze()

depth_image = (
    predicted_depth
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

image_np = np.asarray(image, dtype=np.uint8)

depth_image = np.ascontiguousarray(depth_image)
image_np = np.ascontiguousarray(image_np)

print("RGB shape:", image_np.shape)
print("Depth shape:", depth_image.shape)
print("Depth min:", float(depth_image.min()))
print("Depth max:", float(depth_image.max()))

assert depth_image.shape == image_np.shape[:2]

# ==========================================
# 4. Display RGB and depth map
# ==========================================

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_image, cmap="inferno")
plt.title("Depth Anything V2")
plt.colorbar(label="Relative depth value")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# 5. Create RGB-D image
# ==========================================

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(image_np),
    o3d.geometry.Image(depth_image),
    depth_scale=1.0,
    depth_trunc=10.0,
    convert_rgb_to_intensity=False
)

# ==========================================
# 6. Camera intrinsics (assumed values)
# ==========================================

height, width = depth_image.shape

fx = 700.0
fy = 600.0
cx = (width - 1) / 2.0
cy = (height - 1) / 2.0

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width,
    height,
    fx,
    fy,
    cx,
    cy
)

# ==========================================
# 7. Generate original point cloud
# ==========================================

pcd_original = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    camera_intrinsic
)

print("Number of points:", len(pcd_original.points))

# ==========================================
# 8. Create a copy and flip Y axis
# ==========================================

pcd_flip_y = o3d.geometry.PointCloud(pcd_original)

transform = np.eye(4)
transform[1, 1] = -1

pcd_flip_y.transform(transform)

# ==========================================
# 9. Display point cloud
# ==========================================
o3d.visualization.draw_geometries(
    [pcd_flip_y],
    window_name="Point Cloud - Flip Y",
    width=1280,
    height=800
)
