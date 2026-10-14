
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

from PIL import Image
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
model = model.to(device).eval()

print("Device:", device)

# ==========================================
# 2. Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 3. Initialize Open3D visualizer
# ==========================================

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="Real-time Point Cloud",
    width=1280,
    height=800
)

pcd = o3d.geometry.PointCloud()
geometry_added = False

render_option = vis.get_render_option()
render_option.background_color = np.array([0.05, 0.05, 0.05])
render_option.point_size = 2.0

# ==========================================
# 4. Main loop
# ==========================================

try:
    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            print("Cannot read frame from webcam")
            break

        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        height, width = frame_rgb.shape[:2]

        # --------------------------------------
        # 4.1 Predict depth
        # --------------------------------------

        image_pil = Image.fromarray(frame_rgb)

        inputs = processor(
            images=image_pil,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        predicted_depth = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        depth_image = (
            predicted_depth
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # --------------------------------------
        # 4.2 Display RGB and depth map
        # --------------------------------------

        depth_display = cv2.normalize(
            depth_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(
            depth_display,
            cv2.COLORMAP_INFERNO
        )

        cv2.imshow("Webcam RGB", frame_bgr)
        cv2.imshow("Predicted Depth", depth_colormap)

        # --------------------------------------
        # 4.3 Create RGB-D image
        # --------------------------------------

        depth_image = np.ascontiguousarray(depth_image)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame_rgb),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )

        # --------------------------------------
        # 4.4 Camera intrinsics
        # --------------------------------------

        fx = 700.0
        fy = 600.0
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            fx,
            fy,
            cx,
            cy
        )

        # --------------------------------------
        # 4.5 Generate point cloud
        # --------------------------------------

        pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic
        )

        # Flip Y axis
        transform = np.eye(4)
        transform[1, 1] = -1.0
        pcd_new.transform(transform)


        pcd.points = pcd_new.points
        pcd.colors = pcd_new.colors

        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True
        else:
            vis.update_geometry(pcd)


        vis.poll_events()
        vis.update_renderer()


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
