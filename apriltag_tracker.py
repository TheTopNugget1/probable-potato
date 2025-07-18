import sys
import platform
import cv2
import numpy as np
import yaml
import serial
import time
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from pupil_apriltags import Detector

def setup_serial(port, baud=115200):
    try:
        link = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"Serial connected on {port}")
        return link
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def robust_serial_connect(port, baud=115200, retries=5, delay=2):
    for attempt in range(retries):
        link = setup_serial(port, baud)
        if link and link.is_open:
            return link
        print(f"[Serial] Retry {attempt+1}/{retries} in {delay}s...")
        time.sleep(delay)
    print("[Serial] Failed to connect after retries.")
    return None

def close_serial(link):
    if link and link.is_open:
        link.close()
        print("Serial port closed.")

def send_serial_command(link, command, platform_key, serial_delay=0.01):
    try:
        if not link or not link.is_open:
            print("[Serial] Link not open.")
            return
        if platform_key == "windows":
            link.write((command + "\r\n").encode('utf-8'))
        else:
            link.write((command + "\n").encode('utf-8'))
        print(f"[Serial] Sent: {command.strip()}")
        with open("serial_log.txt", "a") as logf:
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {command.strip()}\n")
        time.sleep(serial_delay)
    except Exception as e:
        print(f"[Serial Error] Failed to send command: {e}")

def send_tag_cord_command_single_line(x: float, y: float, z: float, link, platform_key="windows", serial_delay=0.01):
    if not link or not link.is_open:
        return
    coords = [x, y, z]
    angles = []
    for value in coords:
        value = max(-0.5, min(0.5, value))
        angle = int((value + 0.5) * 360)
        angle = max(0, min(360, angle))
        angles.append(angle)
    command = f"1:{angles[0]} 2:{angles[1]} 3:{angles[2]}"
    send_serial_command(link, command, platform_key, serial_delay)

def load_config(config_path, platform_key):
    print(f"Loading config from {config_path} for platform {platform_key}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    platform_config = config.get(platform_key)
    print(f"Platform config: {platform_config}")

    if not platform_config:
        raise KeyError(f"No config for platform: {platform_key}")
    required_keys = ["tag_id", "tag_size", "tag_family", "camera_matrix", "distortion_coefficients"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        # Allow 0 as valid, only None or empty string is invalid
        if config[key] is None or (isinstance(config[key], str) and config[key].strip() == ""):
            raise ValueError(f"Config key {key} is None or empty!")

    # For camera_id and serial_port, allow 0 and non-empty string
    if "camera_id" not in platform_config:
        raise KeyError(f"Missing camera_id for platform: {platform_key}")
    if platform_config["camera_id"] is None:
        raise ValueError(f"camera_id for platform {platform_key} is None!")

    if "serial_port" not in platform_config:
        raise KeyError(f"Missing serial_port for platform: {platform_key}")
    if platform_config["serial_port"] is None or (isinstance(platform_config["serial_port"], str) and platform_config["serial_port"].strip() == ""):
        raise ValueError(f"serial_port for platform {platform_key} is None or empty!")

    return config, platform_config

def get_aspect_scaled_size(label_width, label_height, aspect_w=4, aspect_h=3):
    # Returns (new_width, new_height) that fits in label and keeps aspect ratio
    if label_width / aspect_w < label_height / aspect_h:
        new_w = label_width
        new_h = int(label_width * aspect_h / aspect_w)
    else:
        new_h = label_height
        new_w = int(label_height * aspect_w / aspect_h)
    return new_w, new_h

class AprilTagTracker(QWidget):
    MODES = ["Joystick", "Hand Tracking"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AprilTag Tracker - PyQt5 UI")
        self.resize(1000, 700)
        # Platform detection
        system = platform.system().lower()
        self.platform_key = "windows" if system == "windows" else "linux"
        # Load config 
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config, self.platform_config = load_config(config_path, self.platform_key)
        self.VALID_TAG_IDS = self.config.get("valid_tag_ids", [])
        self.TAG_ID = self.config["tag_id"]
        self.TAG_SIZE = self.config["tag_size"]
        self.TAG_FAMILY = self.config["tag_family"]
        self.CAMERA_ID = self.platform_config.get("camera_id")
        self.SERIAL_PORT = self.platform_config.get("serial_port")
        self.CAMERA_MATRIX = np.array(self.config["camera_matrix"], dtype=np.float64)
        self.DIST_COEFFS = np.array(self.config["distortion_coefficients"], dtype=np.float64)
        self.SERIAL_DELAY = self.platform_config.get("serial_delay", 0.01)
        self.TAG_ROLES = self.config.get("tag_roles", {})  # {id: "base"/"head"/"cont"}

        # State
        self.serial_link = None
        self.detector = Detector(families=self.TAG_FAMILY)
        self.cap = cv2.VideoCapture(self.CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera failed to open with ID: {self.CAMERA_ID}")
        self.last_sent_coords = [None, None, None]
        self.calibrated = False
        self.calibration_z = 0.0

        # Multi-tag state
        self.detected_tags = {}
        self.base_positions = []      # List of np.array for all base tags
        self.base_center_2d_list = [] # List of 2D centers for all base tags
        self.base_position = None     # Average of base_positions
        self.base_center_2d = None    # Average of base_center_2d_list
        self.head_position = None
        self.head_center_2d = None
        self.rel_head_pos = None
        self.cont_position = None
        self.cont_center_2d = None
        self.rel_cont_pos = None  # this may be subject ot change as it may become obsolete 

        # Target/velocity system
        self.target_position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.last_update_time = time.time()

        # UI Elements
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        self.status_label = QLabel("Serial: Disconnected")
        self.tag_label = QLabel("Tag: Not detected")
        self.connect_btn = QPushButton("Connect Serial")
        self.calib_btn = QPushButton("Calibrate Z")
        self.exit_btn = QPushButton("Exit")

        # Joystick and manual control elements
        self.joystick = JoystickWidget()
        self.z_up_btn = QPushButton("Z+")
        self.z_down_btn = QPushButton("Z-")

        # Overlay: child of video_label
        self.overlay = OverlayWidget(self.joystick, self.z_up_btn, self.z_down_btn, self.video_label)
        self.overlay.setParent(self.video_label)
        self.overlay.setFixedSize(150, 200)
        self.overlay.show()

        # Mode handling
        self.mode_idx = 0  # 0=Joystick, 1=Hand Tracking
        self.mode_toggle_btn = QPushButton(f"Mode: {self.MODES[self.mode_idx]}")

        # Controls layout (right side)
        controls = QVBoxLayout()
        controls.addWidget(self.connect_btn)
        controls.addWidget(self.calib_btn)
        controls.addWidget(self.exit_btn)
        controls.addSpacing(20)
        controls.addWidget(self.status_label)
        controls.addWidget(self.tag_label)
        controls.addWidget(self.mode_toggle_btn)
        controls.addStretch()

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addLayout(controls)
        self.setLayout(main_layout)

        # Signals
        self.connect_btn.clicked.connect(self.handle_connect)
        self.calib_btn.clicked.connect(self.handle_calibrate)
        self.exit_btn.clicked.connect(self.close)
        self.joystick.positionChanged.connect(self.handle_joystick)
        self.z_up_btn.clicked.connect(self.handle_z_up)
        self.z_down_btn.clicked.connect(self.handle_z_down)
        self.mode_toggle_btn.clicked.connect(self.handle_mode_toggle)

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def handle_connect(self):
        if self.serial_link and self.serial_link.is_open:
            close_serial(self.serial_link)
            self.serial_link = None
            self.status_label.setText("Serial: Disconnected")
            self.connect_btn.setText("Connect Serial")
        else:
            self.serial_link = robust_serial_connect(self.SERIAL_PORT)
            if self.serial_link:
                self.status_label.setText("Serial: Connected")
                self.connect_btn.setText("Disconnect Serial")
            else:
                self.status_label.setText("Serial: Failed to connect")

    def handle_calibrate(self):
        if self.tag_position is not None:
            self.calibration_z = self.tag_position[2]
            self.calibrated = True
            QMessageBox.information(self, "Calibration", f"Calibrated Z set to {self.calibration_z:.2f}")
        else:
            QMessageBox.warning(self, "Calibration", "No tag detected for calibration.")

    def handle_mode_toggle(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self.mode_toggle_btn.setText(f"Mode: {self.MODES[self.mode_idx]}")
        self.velocity[:] = 0

    def handle_joystick(self, x, y):
        # Joystick always acts as velocity control for the target variable
        self.velocity[0] = x * 0.05
        self.velocity[1] = y * -0.05

    def handle_z_up(self):
        self.velocity[2] = 0.05

    def handle_z_down(self):
        self.velocity[2] = -0.05

    def update_frame(self):
        # Auto-reconnect serial if lost
        if self.serial_link and not self.serial_link.is_open:
            print("[Serial] Lost connection. Attempting to reconnect...")
            self.serial_link = robust_serial_connect(self.SERIAL_PORT)

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)
        self.detected_tags.clear()
        self.base_positions.clear()
        self.base_center_2d_list.clear()
        self.base_position = None
        self.base_center_2d = None
        self.head_position = None
        self.head_center_2d = None
        self.rel_head_pos = None
        self.cont_position = None
        self.cont_center_2d = None
        self.rel_cont_pos = None
        self.target_marker_2d = None

        # --- Multi-tag tracking and classification ---
        for tag in tags:

            tag_id = tag.tag_id
            role = self.TAG_ROLES.get(str(tag_id), None)

            obj_pts = np.array([
                [-self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0],
                [ self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0],
                [ self.TAG_SIZE / 2,  self.TAG_SIZE / 2, 0],
                [-self.TAG_SIZE / 2,  self.TAG_SIZE / 2, 0]
            ], dtype=np.float32)

            img_pts = np.array(tag.corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.CAMERA_MATRIX, self.DIST_COEFFS)

            if not success:
                continue

            tvec = tvec.flatten()

            if np.linalg.norm(tvec) > 10:
                continue

            # Project center
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            center_2d = tuple(center_2d[0][0].astype(int))
            self.detected_tags[tag_id] = {
                "role": role, "tvec": tvec, "rvec": rvec, "center_2d": center_2d
            }

            # Draw tag corners
            for corner in tag.corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

            # Draw axes
            axis = np.float32([
                [0, 0, 0],
                [0.05, 0, 0],
                [0, 0.05, 0],
                [0, 0, -0.05]
            ])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)
            
            # Draw role label
            if role:
                cv2.putText(frame, f"{role.upper()} [{tag_id}]", (center_2d[0]-30, center_2d[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # Collect base/head/cont positions
            if role == "base":
                self.base_positions.append(tvec)
                self.base_center_2d_list.append(center_2d)
            elif role == "head":
                self.head_position = tvec
                self.head_center_2d = center_2d
            elif role == "cont":
                self.cont_position = tvec
                self.cont_center_2d = center_2d

        # --- Calculate average base position (multi-tag base localization) ---
        if self.base_positions:
            self.base_position = np.mean(self.base_positions, axis=0)
            self.base_center_2d = tuple(np.mean(self.base_center_2d_list, axis=0).astype(int))
        else:
            self.base_position = None
            self.base_center_2d = None

        # --- Compute relative positions ---
        if self.base_position is not None and self.head_position is not None:
            self.rel_head_pos = self.head_position - self.base_position
        if self.base_position is not None and self.cont_position is not None:
            self.rel_cont_pos = self.cont_position - self.base_position

        # --- Target position update ---
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        if self.mode_idx == 0:  # Joystick mode
            self.target_position += self.velocity * dt
            self.target_position = np.clip(self.target_position, -0.5, 0.5)
            self.velocity[2] = 0
            target = self.target_position
            self.tag_label.setText(f"Joystick: X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f}")
        elif self.mode_idx == 1:  # Hand Tracking mode
            if self.rel_cont_pos is not None:
                target = self.rel_cont_pos
                self.tag_label.setText(f"Hand: X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f}")
            else:
                target = None
                self.tag_label.setText("Hand: Controller or base not detected")
        else:
            target = None

        # --- Target marker overlay ---
        if self.base_position is not None and target is not None:
            tvec = self.base_position + target
            rvec = np.zeros((3,1), dtype=np.float32)
            marker_3d = np.array([[target[0], target[1], target[2]]], dtype=np.float32)
            marker_2d, _ = cv2.projectPoints(marker_3d, rvec, self.base_position, self.CAMERA_MATRIX, self.DIST_COEFFS)
            self.target_marker_2d = tuple(marker_2d[0][0].astype(int))
            cv2.drawMarker(frame, self.target_marker_2d, (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(frame, "TARGET", (self.target_marker_2d[0]+10, self.target_marker_2d[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        else:
            self.target_marker_2d = None

        # --- Overlay: draw line base-head, head XYZ rel to base ---
        if self.base_center_2d and self.head_center_2d:
            cv2.line(frame, self.base_center_2d, self.head_center_2d, (0,255,255), 2)
        if self.head_center_2d and self.rel_head_pos is not None:
            x, y, z = self.rel_head_pos
            cv2.putText(frame, f"Head rel: X={x:.3f} Y={y:.3f} Z={z:.3f}",
                        (self.head_center_2d[0]+10, self.head_center_2d[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # --- 3D Message Box Overlay for base ---
        if self.base_center_2d is not None:
            box_w, box_h = 120, 40
            x, y = self.base_center_2d
            box_x = x - box_w // 2
            box_y = y - box_h - 10
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
            cv2.putText(frame, "BASE", (box_x + 10, box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # --- Serial/label logic (can be expanded later) ---
        # (No sending to Arduino for now, as per your request)

        # Show serial status
        status = "Connected" if self.serial_link and self.serial_link.is_open else "Disconnected"
        self.status_label.setText(f"Serial: {status}")

        # Draw overlays
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        aspect_w, aspect_h = 4, 3  # Change to 16, 9 for 16:9

        # Calculate the largest size that fits and keeps aspect ratio
        new_w, new_h = get_aspect_scaled_size(label_width, label_height, aspect_w, aspect_h)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Center the pixmap in the label if there's extra space
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

        # --- Inverse Kinematics Output (for documentation/debugging) ---
        if target is not None and self.base_position is not None:
            ik_result = self.compute_ik_3dof(target, link_lengths=(0.2, 0.2))  # Adjust link_lengths as needed
            if ik_result is not None:
                print(f"Desired joint angles (deg): {ik_result}")
            else:
                print("No valid IK solution for this target.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.serial_link:
            close_serial(self.serial_link)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Center overlay halfway up the left side of the video_label
        video_h = self.video_label.height()
        overlay_h = self.overlay.height()
        x = 0  # left edge
        y = (video_h - overlay_h) // 2
        self.overlay.move(x, y)

    def compute_ik_3dof(self, target, link_lengths=(0.2, 0.2)):
        """
        3DOF arm: base rotation + 2 planar links (shoulder, elbow)
        target: np.array([x, y, z])
        link_lengths: tuple of (l1, l2)
        Returns (theta0_deg, theta1_deg, theta2_deg) or None if unreachable.
        """
        x, y, z = target
        l1, l2 = link_lengths
        # 1. Base rotation
        theta0 = np.arctan2(y, x)
        # 2. Project into arm's plane
        r = np.sqrt(x**2 + y**2)
        # 3. Planar IK in (r, z)
        D = (r**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(D) > 1:
            print("Target out of reach for IK.")
            return None
        theta2 = np.arccos(D)
        theta1 = np.arctan2(z, r) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        return np.degrees(theta0), np.degrees(theta1), np.degrees(theta2)

class JoystickWidget(QWidget):
    positionChanged = pyqtSignal(float, float)  # Emits normalized x, y in [-1, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self.center = QPoint(self.width() // 2, self.height() // 2)
        self.knob_pos = self.center
        self.radius = min(self.width(), self.height()) // 2 - 10
        self.active = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw base (semi-transparent)
        painter.setBrush(QColor(220, 220, 220, 120))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.center, self.radius, self.radius)
        # Draw knob (more opaque)
        painter.setBrush(QColor(100, 100, 255, 180))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.knob_pos, 18, 18)

    def mousePressEvent(self, event):
        if (event.pos() - self.center).manhattanLength() <= self.radius:
            self.active = True
            self.update_knob(event.pos())

    def mouseMoveEvent(self, event):
        if self.active:
            self.update_knob(event.pos())

    def mouseReleaseEvent(self, event):
        self.active = False
        self.knob_pos = self.center
        self.positionChanged.emit(0.0, 0.0)
        self.update()

    def update_knob(self, pos):
        dx = pos.x() - self.center.x()
        dy = pos.y() - self.center.y()
        dist = (dx**2 + dy**2) ** 0.5
        if dist > self.radius:
            dx = dx * self.radius / dist
            dy = dy * self.radius / dist
        self.knob_pos = QPoint(self.center.x() + int(dx), self.center.y() + int(dy))
        norm_x = dx / self.radius
        norm_y = -dy / self.radius  # Invert Y for UI
        self.positionChanged.emit(norm_x, norm_y)
        self.update()

class OverlayWidget(QWidget):
    def __init__(self, joystick, z_up_btn, z_down_btn, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(joystick)
        layout.addWidget(z_up_btn)
        layout.addWidget(z_down_btn)
        layout.addStretch()
        self.setLayout(layout)
        # Set transparency for child widgets
        joystick.setStyleSheet("background: transparent;")
        z_up_btn.setStyleSheet("background: rgba(255,255,255,120);")
        z_down_btn.setStyleSheet("background: rgba(255,255,255,120);")

    def paintEvent(self, event):
        # Optional: draw a transparent background for the overlay itself
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(255, 255, 255, 40))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AprilTagTracker()
    window.show()
    sys.exit(app.exec_())
