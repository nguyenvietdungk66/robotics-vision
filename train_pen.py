import cv2
import numpy as np
import os
import platform
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# Tự động chọn đường dẫn theo hệ điều hành
if platform.system() == 'Windows':
    BASE = 'D:/Coding'
else:  # Linux / Ubuntu
    BASE = '/media/hung/New Volume/Coding'

# Model YOLO26 đã được train (từ yolo26n.pt trên dataset bút)
MODEL_PATH  = os.path.join(BASE, 'runs/detect/train-3/weights/best.pt')
TEST_IMAGE  = os.path.join(BASE, 'LAB1/pen detection/pen2.jpg')
CALIB_PATH  = os.path.join(BASE, 'LAB1/calibration.npz')
OUTPUT_PATH = os.path.join(BASE, 'LAB1/depth_result.jpg')
CONF_THRESH = 0.45



def loadCalibration(path):
    if not os.path.exists(path):
        print(f"[!] Không tìm thấy {path} — dùng camMatrix mặc định")
        fx = fy = 1000.0
        cx, cy  = 540.0, 960.0
        camMatrix = np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]], dtype=np.float64)
        distCoeffs = np.zeros((5,))
    else:
        data       = np.load(path)
        camMatrix  = data['camMatrix']
        distCoeffs = data['distCoeff'] if 'distCoeff' in data else data['distCoeffs']
        print(f"[✓] Đã load calibration: fx={camMatrix[0,0]:.1f}, fy={camMatrix[1,1]:.1f}")
    return camMatrix, distCoeffs



def loadAndUndistort(img_path, camMatrix, distCoeffs):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    h, w = img.shape[:2]
    newCamMatrix, _ = cv2.getOptimalNewCameraMatrix(camMatrix, distCoeffs, (w, h), 1, (w, h))
    img_undist = cv2.undistort(img, camMatrix, distCoeffs, None, newCamMatrix)
    return img_undist, newCamMatrix  



def detectObjects(img, model_path, conf=0.45):
    print("\n[YOLO26] Đang nhận diện vật thể...")
    # YOLO26 dùng cùng API ultralytics với YOLOv8
    # Chỉ cần đổi file .pt — không cần thay đổi cách gọi
    yolo    = YOLO(model_path)
    results = yolo.predict(source=img, conf=conf, show=False, save=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({'label': r.names[cls],'conf' : conf,'box'  : (int(x1), int(y1), int(x2), int(y2)),'centroid': (cx, cy)})
            print(f"  [✓] {r.names[cls]} | conf={conf:.1%} | box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) | centroid=({cx},{cy})")

    if not detections:
        print("  [!] Không tìm thấy vật thể nào.")
    return detections

#  Dùng bản METRIC thay vì bản tương đối:
#    - Bản tương đối (Small-hf): trả về 0.0→1.0, không có đơn vị
#    - Bản metric (Metric-Indoor): trả về mét thực tế → dùng trực tiếp
#  → Không cần vật tham chiếu cố định, không cần scale calibration

def estimateDepth(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    depth_estimator = pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

    depth_result = depth_estimator(pil_img)
    import torch
    raw = depth_result["predicted_depth"]
    if isinstance(raw, torch.Tensor):
        depth_map = raw.squeeze().numpy()   # bỏ chiều batch nếu có: (1,H,W)→(H,W)
    else:
        depth_map = np.array(raw).squeeze()

    depth_resized = cv2.resize(depth_map.astype(np.float32),(img_bgr.shape[1], img_bgr.shape[0]))

    print(f"  [✓] Depth map: {depth_resized.shape} | Range: {depth_resized.min():.2f}m – {depth_resized.max():.2f}m")
    if depth_resized.max() > 20:
        print("  [!] Depth > 20m — kiểm tra lại model có phải bản Metric không")
    return depth_resized


#  Từ pixel (u,v) + depth Z (mét) + camMatrix → (X,Y,Z) camera frame
#    X_cam = (u - cx) * Z / fx
#    Y_cam = (v - cy) * Z / fy
#    Z_cam = Z  
def pixelToCamera3D(u, v, Z_meters, camMatrix):
    fx = camMatrix[0, 0]
    fy = camMatrix[1, 1]
    cx = camMatrix[0, 2]
    cy = camMatrix[1, 2]

    X_cam = (u - cx) * Z_meters / fx
    Y_cam = (v - cy) * Z_meters / fy
    Z_cam = Z_meters

    return X_cam, Y_cam, Z_cam
# ──────────────────────────────────────────────────────────────────────
#  PHÂN VÙNG BÚT: WATERSHED → GRABCUT
#
#  Quy trình:
#    1. Bilateral Filter → Threshold → Morphology → Sure Background
#    2. Distance Transform → Sure Foreground (lõi vật thể)
#    3. Unknown = Sure_BG - Sure_FG
#    4. Connected Components → Watershed Markers
#    5. cv2.watershed() → Biên chính xác
#    6. Chuyển markers sang GrabCut mask (GC_BGD/GC_FGD/GC_PR_FGD/GC_PR_BGD)
#    7. cv2.grabCut(GC_INIT_WITH_MASK) → Mask tinh chỉnh cuối cùng
#
#  Tại sao cần cả hai?
#    - Watershed: phân vùng dựa trên gradient, tách biên tốt nhưng dễ over-segment
#    - GrabCut: mô hình GMM tinh chỉnh pixel-level, cần seed tốt (từ Watershed)
#    → Kết hợp: Watershed cung cấp seed chất lượng cho GrabCut
# ──────────────────────────────────────────────────────────────────────

def segmentPenWatershedGrabCut(roi_bgr):
    """Phân vùng bút bằng Watershed + GrabCut.
    Watershed tạo markers (sure_fg/sure_bg/unknown),
    sau đó chuyển thành mask cho GrabCut tinh chỉnh.
    """
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 10:
        return np.zeros((h, w), dtype=np.uint8)

    # === BƯỚC 1: Tiền xử lý bằng Bilateral Filter (bảo toàn biên) ===
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # === BƯỚC 2: Threshold → tìm vùng foreground thô ===
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # === BƯỚC 3: Morphology → Sure Background ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # === BƯỚC 4: Distance Transform → Sure Foreground ===
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    if dist_max < 1.0:
        # ROI quá nhỏ hoặc threshold thất bại → fallback
        print("    [Watershed] Distance transform quá phẳng → fallback sang Canny")
        return np.zeros((h, w), dtype=np.uint8)

    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_max, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # === BƯỚC 5: Unknown = Sure_BG - Sure_FG ===
    unknown = cv2.subtract(sure_bg, sure_fg)

    # === BƯỚC 6: Watershed Markers ===
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # Background = 1 (không phải 0)
    markers[unknown == 255] = 0    # Unknown = 0 (chưa biết)

    # Chạy Watershed trên ảnh BGR
    markers = np.int32(markers)
    markers = cv2.watershed(roi_bgr, markers)
    # markers == -1 là biên, markers == 1 là background, markers > 1 là foreground

    # === BƯỚC 7: Chuyển Watershed markers → GrabCut mask ===
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Viền ngoài cùng (6%) → chắc chắn nền
    pad_h = max(2, int(h * 0.06))
    pad_w = max(2, int(w * 0.06))
    gc_mask[:pad_h, :] = cv2.GC_BGD
    gc_mask[-pad_h:, :] = cv2.GC_BGD
    gc_mask[:, :pad_w] = cv2.GC_BGD
    gc_mask[:, -pad_w:] = cv2.GC_BGD

    # Watershed background (marker == 1) → GC_BGD
    gc_mask[markers == 1] = cv2.GC_BGD
    # Watershed foreground (marker > 1) → GC_PR_FGD
    gc_mask[markers > 1] = cv2.GC_PR_FGD
    # Watershed biên (marker == -1) → GC_PR_BGD (để GrabCut quyết định)
    gc_mask[markers == -1] = cv2.GC_PR_BGD

    # Lõi sâu nhất của distance transform → GC_FGD (chắc chắn foreground)
    core_fg = (dist_transform > 0.55 * dist_max)
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[pad_h:h-pad_h, pad_w:w-pad_w] = True
    gc_mask[core_fg & inner_mask] = cv2.GC_FGD

    # === BƯỚC 8: GrabCut tinh chỉnh ===
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        pen_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception as e:
        print(f"    [GrabCut] Fallback: {e}")
        pen_mask = np.where(markers > 1, 255, 0).astype(np.uint8)

    # === BƯỚC 9: Hậu xử lý — giữ thành phần liên thông lớn nhất ===
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pen_mask = cv2.morphologyEx(pen_mask, cv2.MORPH_CLOSE, k_clean, iterations=2)
    pen_mask = cv2.morphologyEx(pen_mask, cv2.MORPH_OPEN, k_clean, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pen_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pen_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    fg_pct = 100 * np.count_nonzero(pen_mask) / pen_mask.size
    print(f"    [Watershed+GrabCut] Mask: {np.count_nonzero(pen_mask)} px ({fg_pct:.1f}% of ROI)")

    return pen_mask


#  Phương pháp kết hợp:
#    A) Ellipse Fitting từ contour → Yaw (góc nghiêng trong mặt phẳng ảnh)
#                                  → Pitch (từ tỉ lệ trục b/a của ellipse)
#    B) Depth Map Gradient → Roll, Pitch (hướng pháp tuyến mặt phẳng cục bộ)
#  Hệ quy chiếu góc (camera frame):
#    - Yaw   (ψ): Xoay quanh trục Z (trục độ sâu) — góc bút nằm ngang/dọc
#    - Pitch (θ): Xoay quanh trục X (trái-phải) — bút nghiêng xa/gần camera
#    - Roll  (φ): Xoay quanh trục Y (lên-xuống) — bút nghiêng trái/phải
def estimateOrientation(img, depth_resized, detection, camMatrix):
    x1, y1, x2, y2 = detection['box']
    cx_obj, cy_obj = detection['centroid']
    pad = 10
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(img.shape[1], x2 + pad)
    roi_y2 = min(img.shape[0], y2 + pad)

    result = {'yaw_deg'  : 0.0,'pitch_deg': 0.0,'roll_deg' : 0.0,'rotation_matrix': np.eye(3),'ellipse_angle'  : 0.0,'axes_ratio'     : 1.0,}
    #  A) ELLIPSE FITTING → Yaw + Pitch ước lượng
    #  Logic:
    #    1. Lấy ROI từ bounding box, chuyển grayscale
    #    2. Threshold + Canny → contour viền bút
    #    3. fitEllipse() → (center2D, (MA, ma), angle)
    #       - angle: góc trục dài của ellipse so với trục ngang (độ)
    #         → đây chính là Yaw ảo trong 2D
    #       - MA (Major Axis), ma (minor axis)
    #       - ratio = ma / MA: nếu = 1 → vật nhìn thẳng đầu (pitch≈0°)
    #                           nếu = 0 → vật nằm ngang tuyệt đối (pitch≈90°)
    # ──────────────────────────────────────────────────────────
    roi_img  = img[roi_y1:roi_y2, roi_x1:roi_x2]

    # ── Phân vùng bằng Watershed + GrabCut ──
    pen_mask = segmentPenWatershedGrabCut(roi_img)

    # Nếu phân vùng thành công → lấy contour từ mask
    # Nếu thất bại (mask rỗng) → fallback sang Bilateral + Canny
    if np.count_nonzero(pen_mask) > 100:
        contours, _ = cv2.findContours(pen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("    [✓] Dùng contour từ Watershed+GrabCut mask")
    else:
        print("    [!] Mask rỗng → fallback: Bilateral Filter + Canny")
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.bilateralFilter(gray_roi, d=9, sigmaColor=75, sigmaSpace=75)
        edges    = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (ex, ey), (MA, ma), angle = ellipse

            result['ellipse_angle'] = angle
            result['yaw_deg'] = angle - 90.0 

            # Tỉ lệ trục: ma/MA ∈ [0, 1]
            #   → Nếu gần 1: bút nhìn từ đầu (tròn) → pitch ≈ 0° (thẳng vào camera)
            #   → Nếu gần 0: bút nằm ngang phẳng → pitch ≈ 90°
            # Nhưng với bút dài: nếu nhìn nghiêng thì ellipse trở nên tròn hơn
            # Công thức: pitch = arccos(ma / MA)  (xấp xỉ)
            MA_safe = MA if MA > 1e-3 else 1e-3
            axes_ratio = np.clip(ma / MA_safe, 0.0, 1.0)
            result['axes_ratio'] = axes_ratio
            pitch_rad = np.arccos(axes_ratio)  
            result['pitch_deg'] = np.degrees(pitch_rad)

            print(f"  [Ellipse] angle={angle:.1f}° | MA={MA:.1f}px, ma={ma:.1f}px | ratio={axes_ratio:.3f}")
            print(f"  → Yaw ước lượng  : {result['yaw_deg']:.1f}°")
            print(f"  → Pitch ước lượng: {result['pitch_deg']:.1f}°")
        else:
            print(f"  [Ellipse] Contour quá nhỏ ({len(largest_contour)} pts) — bỏ qua")
    else:
        print("  [Ellipse] Không tìm được contour trong bounding box")

    # ──────────────────────────────────────────────────────────
    #  B) DEPTH MAP GRADIENT → Roll ước lượng
    #  Logic:
    #    Lấy vùng depth map trong bounding box → tính gradient ∂Z/∂x, ∂Z/∂y
    #    Surface normal n = (-∂Z/∂x, -∂Z/∂y, 1) (xấp xỉ cho mặt phẳng cục bộ)
    #    Roll = góc giữa normal vector và trục Z của camera
    #
    #  Lưu ý: Depth Anything V2 cho depth mượt → gradient noise có thể cao
    #  → dùng Sobel kernel kích thước lớn + làm mịn trước
    # ──────────────────────────────────────────────────────────
    depth_roi = depth_resized[roi_y1:roi_y2, roi_x1:roi_x2]

    if depth_roi.size > 0 and depth_roi.std() > 1e-4:
        depth_smooth = cv2.GaussianBlur(depth_roi, (7, 7), 2)
        gx = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=5)


        mean_gx = float(np.mean(gx))
        mean_gy = float(np.mean(gy))


        normal = np.array([-mean_gx, -mean_gy, 1.0])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len

        roll_rad = np.arctan2(normal[0], normal[2])
        result['roll_deg'] = np.degrees(roll_rad)

        print(f"  [Depth Grad] gx={mean_gx:.4f}, gy={mean_gy:.4f}")
        print(f"  [Depth Grad] Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
        print(f"  → Roll ước lượng : {result['roll_deg']:.1f}°")
    else:
        print("  [Depth Grad] Depth ROI quá phẳng — Roll = 0°")

    #  Xây dựng Rotation Matrix từ Euler angles (ZYX convention)
    #  Thứ tự: Yaw (Z) → Pitch (Y) → Roll (X)
    #  Dùng để vẽ trục tọa độ 3D lên ảnh bằng cv2.projectPoints
    roll  = np.radians(result['roll_deg'])
    pitch = np.radians(result['pitch_deg'])
    yaw   = np.radians(result['yaw_deg'])

    # Ma trận xoay cơ bản
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(roll), -np.sin(roll)],
                   [0,  np.sin(roll),  np.cos(roll)]])

    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1,             0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    result['rotation_matrix'] = Rz @ Ry @ Rx

    return result

#  Cần Hand-Eye Calibration để có ma trận T_cam2robot (4x4).
#  Khi có T_cam2robot rồi thì:
#    P_robot = T_cam2robot @ [X_cam, Y_cam, Z_cam, 1]
#    R_robot = T_cam2robot[:3,:3] @ R_cam
#
#  Hiện tại trả về tọa độ camera vì chưa có T_cam2robot.
def cameraToRobot(X_cam, Y_cam, Z_cam, R_cam=None, T_cam2robot=None):
    if T_cam2robot is None:
        print("  [!] Chưa có T_cam2robot — trả về tọa độ camera frame")
        return X_cam, Y_cam, Z_cam, R_cam

    P_cam   = np.array([X_cam, Y_cam, Z_cam, 1.0])
    P_robot = T_cam2robot @ P_cam

    R_robot = None
    if R_cam is not None:
        R_robot = T_cam2robot[:3, :3] @ R_cam

    return P_robot[0], P_robot[1], P_robot[2], R_robot


def draw3DAxes(img_draw, cx, cy, Z_meters, rotation_matrix, camMatrix,axis_len_m=0.05):

    fx = camMatrix[0, 0]
    fy = camMatrix[1, 1]
    cx_cam = camMatrix[0, 2]
    cy_cam = camMatrix[1, 2]

    origin_3d = np.array([(cx - cx_cam) * Z_meters / fx,(cy - cy_cam) * Z_meters / fy,Z_meters])

    axis_pts_local = np.array([[axis_len_m, 0, 0],[0, axis_len_m, 0],[0, 0, -axis_len_m]])
    axis_pts_cam = (rotation_matrix @ axis_pts_local.T).T + origin_3d

    def project(pt3d):
        if abs(pt3d[2]) < 1e-6:
            return None
        u = int(fx * pt3d[0] / pt3d[2] + cx_cam)
        v = int(fy * pt3d[1] / pt3d[2] + cy_cam)
        return (u, v)

    origin_2d = (cx, cy)
    colors = [(0, 0, 225), (0, 225, 0), (225, 0, 0)]  
    labels = ['X', 'Y', 'Z']
    thickness = 3
    tip_r = 6

    for i, (pt3d, color, label) in enumerate(zip(axis_pts_cam, colors, labels)):
        tip_2d = project(pt3d)
        if tip_2d is None:
            continue
        h_img, w_img = img_draw.shape[:2]
        if not (0 <= tip_2d[0] < w_img and 0 <= tip_2d[1] < h_img):
            continue
        cv2.arrowedLine(img_draw, origin_2d, tip_2d, color, thickness, tipLength=0.3)
        cv2.putText(img_draw, label, (tip_2d[0] + 4, tip_2d[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def visualize(img, depth_resized, detections, coords_3d, orientations, camMatrix):
    depth_norm    = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

    img_draw = img.copy()

    for det, coord, orient in zip(detections, coords_3d, orientations):
        x1, y1, x2, y2 = det['box']
        cx, cy = det['centroid']
        X, Y, Z = coord
        roll  = orient['roll_deg']
        pitch = orient['pitch_deg']
        yaw   = orient['yaw_deg']
        R_mat = orient['rotation_matrix']

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 230, 0), 2)

        cv2.circle(img_draw, (cx, cy), 6, (0, 0, 255), -1)

        label_top = f"{det['label']}  Z={Z:.2f}m  conf={det['conf']:.0%}"
        cv2.putText(img_draw, label_top, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.65, (0, 230, 0), 2)

        cv2.putText(img_draw,
                    f"Pos: ({X:.2f}, {Y:.2f}, {Z:.2f}) m",
                    (x1, y2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 2)
        cv2.putText(img_draw,
                    f"R={roll:.1f}  P={pitch:.1f}  Y={yaw:.1f} deg",
                    (x1, y2 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 180, 0), 2)

        if Z > 0.01:
            draw3DAxes(img_draw, cx, cy, Z, R_mat, camMatrix, axis_len_m=0.04)

        cv2.circle(depth_colored, (cx, cy), 20, (0, 255, 0), -1)
        cv2.putText(depth_colored, f"Z={Z:.2f}m", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 2)
        cv2.putText(depth_colored,
                    f"Y={yaw:.1f}P={pitch:.1f}R={roll:.1f}",
                    (cx + 10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.55, (255, 220, 0), 2)

    legend_y = 30
    for label, color in [('X-axis (Roll dir)', (0, 0, 220)),
                          ('Y-axis (Pitch dir)', (0, 200, 0)),
                          ('Z-axis (Yaw dir)', (220, 100, 0))]:
        cv2.putText(img_draw, label, (12, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.55, color, 2)
        legend_y += 24

    h = img_draw.shape[0]
    depth_colored = cv2.resize(
        depth_colored,
        (int(depth_colored.shape[1] * h / depth_colored.shape[0]), h)
    )
    combined = np.hstack([img_draw, depth_colored])
    return combined
def main():
    print("=" * 60)
    print("  PIPELINE 6-DoF: YOLO11 + Depth Anything V2 + Orientation")
    print("  Output: Position (X,Y,Z) + Rotation (Roll,Pitch,Yaw)")
    print("=" * 60)

    camMatrix, distCoeffs = loadCalibration(CALIB_PATH)

    img, camMatrix = loadAndUndistort(TEST_IMAGE, camMatrix, distCoeffs)
    detections = detectObjects(img, MODEL_PATH, conf=CONF_THRESH)
    if not detections:
        print("\n[!] Không có vật thể → dừng.")
        return

    depth_map = estimateDepth(img)
    print("\n" + "─" * 60)
    print("  KẾT QUẢ 6-DoF")
    print("─" * 60)

    coords_3d    = []
    orientations = []

    for i, det in enumerate(detections):
        print(f"\n  [{i+1}] Vật: {det['label']} | conf={det['conf']:.1%}")
        cx, cy = det['centroid']

        Z = float(depth_map[cy, cx])
        X_cam, Y_cam, Z_cam = pixelToCamera3D(cx, cy, Z, camMatrix)
        X_robot, Y_robot, Z_robot, _ = cameraToRobot(
            X_cam, Y_cam, Z_cam, R_cam=None, T_cam2robot=None
        )
        coords_3d.append((X_robot, Y_robot, Z_robot))

        print(f"  Centroid pixel : ({cx}, {cy})")
        print(f"  Depth (mét)    : {Z:.3f} m")
        print(f"  Tọa độ Camera  : X={X_cam:.3f}m  Y={Y_cam:.3f}m  Z={Z_cam:.3f}m")

        print(f"  Đang ước lượng góc 3D...")
        orient = estimateOrientation(img, depth_map, det, camMatrix)
        orientations.append(orient)

        roll  = orient['roll_deg']
        pitch = orient['pitch_deg']
        yaw   = orient['yaw_deg']

        print(f"\n  ┌─ 6-DoF POSE ──────────────────────────────────┐")
        print(f"  │  POSITION : X={X_robot:.3f}m  Y={Y_robot:.3f}m  Z={Z_robot:.3f}m   │")
        print(f"  │  ROTATION : Roll={roll:+7.1f}°  Pitch={pitch:+7.1f}°  Yaw={yaw:+7.1f}° │")
        print(f"  └────────────────────────────────────────────────┘")
        print(f"  → Gửi robot: MOVE_TO ({X_robot:.3f}, {Y_robot:.3f}, {Z_robot:.3f})")
        print(f"              ROTATE  (R={roll:.1f}°, P={pitch:.1f}°, Y={yaw:.1f}°)")

    combined = visualize(img, depth_map, detections, coords_3d, orientations, camMatrix)
    cv2.imwrite(OUTPUT_PATH, combined)
    print(f"\n[✓] Ảnh kết quả đã lưu tại: {OUTPUT_PATH}", flush=True)
    try:
        cv2.imshow("6-DoF Pose: YOLO11 + Depth Anything V2", combined)
        print("Cửa sổ hiển thị ảnh đã mở (tự đóng sau 3 giây hoặc nhấn phím)...", flush=True)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    except Exception as e:
        pass


if __name__ == '__main__':
    main()
