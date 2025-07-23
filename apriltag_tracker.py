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

def setup_serial(port, baud):
    try:
        link = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"Serial connected on {port}")
        return link
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def robust_serial_connect(port, baud, retries, delay):
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

def send_serial_command(link, command, platform_key, serial_delay):
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
    required_keys = ["tag_size", "tag_family", "camera_matrix", "distortion_coefficients"]

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
        self.TAG_ID = None  # Default to None
        self.TAG_SIZE = self.config["tag_size"]
        self.TAG_FAMILY = self.config["tag_family"]
        self.TAG_ROLES = self.config.get("tag_roles", {})  # {id: "base"/"head"/"cont"}

        self.CAMERA_ID = self.platform_config.get("camera_id")
        self.SERIAL_PORT = self.platform_config.get("serial_port")
        self.BAUD_RATE = self.platform_config.get("baud_rate")
        self.RETRIES = self.platform_config.get("retries")
        self.SETUP_DELAY = self.platform_config.get("setup_delay")

        self.CAMERA_MATRIX = np.array(self.config["camera_matrix"], dtype=np.float64)
        self.DIST_COEFFS = np.array(self.config["distortion_coefficients"], dtype=np.float64)
        self.SERIAL_DELAY = self.platform_config.get("serial_delay")
        
        # State
        self.serial_link = None
        self.detector = Detector(families=self.TAG_FAMILY)
        self.cap = cv2.VideoCapture(self.CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera failed to open with ID: {self.CAMERA_ID}")

        # Multi-tag state
        self.detected_tags = {}

        self.base_positions = []      # List of np.array for all base tags
        self.base_center_2d_list = [] # List of 2D centers for all base tags

        self.base_position = None     # Average of base_positions
        self.base_center_2d = None    # Average of base_center_2d_list

        self.cont_position = None
        self.cont_center_2d = None 
        self.rel_cont_pos = None  # Relative position of cont tag to base

        # Target/velocity system
        self.target_position = np.zeros(3, dtype=np.float32)
        self.last_printed_target = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.z_velocity = 0.0  # For Z-axis joystick control
        self.last_update_time = time.time()

        # UI Elements
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        self.status_label = QLabel("Serial: Disconnected")
        self.tag_label = QLabel("Tag: Not detected")
        self.connect_btn = QPushButton("Connect Serial")
        self.exit_btn = QPushButton("Exit")

        # Joystick and manual control elements
        self.joystick = JoystickWidget()
        self.z_joystick = ZJoystickWidget()
        

        # Overlay: child of video_label
        self.left_overlay = OverlayWidget(self.joystick, None, self.video_label)
        self.left_overlay.setParent(self.video_label)
        self.left_overlay.setFixedSize(120, 120)
        self.left_overlay.show()

        self.right_overlay = OverlayWidget(None, self.z_joystick, self.video_label)
        self.right_overlay.setParent(self.video_label)
        self.right_overlay.setFixedSize(60, 120)
        self.right_overlay.show()

        # Mode handling
        self.mode_idx = 0  # 0=Joystick, 1=Hand Tracking
        self.mode_toggle_btn = QPushButton(f"Mode: {self.MODES[self.mode_idx]}")

        # Controls layout (right side)
        controls = QVBoxLayout()
        controls.addWidget(self.connect_btn)
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
        self.exit_btn.clicked.connect(self.close)
        self.joystick.positionChanged.connect(self.handle_joystick)
        self.z_joystick.zPositionChanged.connect(self.handle_z_joystick)
        self.mode_toggle_btn.clicked.connect(self.handle_mode_toggle)

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        # Position overlays at startup
        QTimer.singleShot(0, self.position_overlays)

    def position_overlays(self):
        # Get video dimensions
        video_w = self.video_label.width()
        video_h = self.video_label.height()
        
        # Position left overlay (regular joystick) on the left side
        left_overlay_h = self.left_overlay.height()
        left_x = 10  # Small margin from left edge
        left_y = (video_h - left_overlay_h) // 2
        self.left_overlay.move(left_x, left_y)
        
        # Position right overlay (Z joystick) on the right side
        right_overlay_w = self.right_overlay.width()
        right_overlay_h = self.right_overlay.height()
        right_x = video_w - right_overlay_w - 10  # Small margin from right edge
        right_y = (video_h - right_overlay_h) // 2
        self.right_overlay.move(right_x, right_y)

    def handle_connect(self):
        if self.serial_link and self.serial_link.is_open:
            close_serial(self.serial_link)
            self.serial_link = None
            self.status_label.setText("Serial: Disconnected")
            self.connect_btn.setText("Connect Serial")
        else:
            self.serial_link = robust_serial_connect(self.SERIAL_PORT, self.BAUD_RATE, self.RETRIES, self.SETUP_DELAY)
            if self.serial_link:
                self.status_label.setText("Serial: Connected")
                self.connect_btn.setText("Disconnect Serial")
            else:
                self.status_label.setText("Serial: Failed to connect")

    def handle_mode_toggle(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self.mode_toggle_btn.setText(f"Mode: {self.MODES[self.mode_idx]}")
        self.velocity[:] = 0

    def handle_joystick(self, x, y):
        # Joystick always acts as velocity control for the target variable
        self.velocity[0] = x * 0.05
        self.velocity[1] = y * -0.05

    def handle_z_joystick(self, z):
        # Update Z-axis velocity based on joystick input
        self.z_velocity = z * 0.05  # Adjust scaling factor as needed
        #self.velocity[2] = z * 0.05  # Adjust scaling factor as needed

    def draw_tag_boxes(self, frame):
        for tag_id, tag_data in self.detected_tags.items():
            if tag_id in self.VALID_TAG_IDS:
                center_2d = tag_data["center_2d"]
                role = tag_data["role"]

                # Draw label on the tag
                cv2.putText(frame, f"{role.upper()} [{tag_id}]", (center_2d[0] - 30, center_2d[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 1)

                # Draw dots around the tag corners
                for corner in tag_data["corners"]:
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

                # Draw axes on the tag
                rvec, tvec = tag_data["rvec"], tag_data["tvec"]
                axis = np.float32([
                    [0, 0, 0],
                    [0.05, 0, 0],
                    [0, 0.05, 0],
                    [0, 0, -0.05]
                ])
                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
                imgpts = np.int32(imgpts).reshape(-1, 2)
                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)  # X-axis (red)
                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)  # Z-axis (blue)

                # base label 
                if self.base_center_2d is not None:
                    box_w, box_h = 60, 20
                    x, y = self.base_center_2d
                    box_x = x - box_w // 2
                    box_y = y - box_h - 10
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
                    cv2.putText(frame, "BASE", (box_x + 5, box_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                
    def update_frame(self):
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
        self.cont_position = None
        self.cont_center_2d = None
        self.target_marker_2d = None

        for tag in tags:
            tag_id = tag.tag_id
            role = self.TAG_ROLES.get(str(tag_id), None)

            obj_pts = np.array([
                [-self.TAG_SIZE / 2, self.TAG_SIZE / 2, 0],
                [self.TAG_SIZE / 2, self.TAG_SIZE / 2, 0],
                [self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0],
                [-self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0]
            ], dtype=np.float32)

            img_pts = np.array(tag.corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.CAMERA_MATRIX, self.DIST_COEFFS)

            if not success:
                continue

            tvec = tvec.flatten()

            if np.linalg.norm(tvec) > 10:
                continue

            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            center_2d = tuple(center_2d[0][0].astype(int))

            self.detected_tags[tag_id] = {
                "role": role, "tvec": tvec, "rvec": rvec, "center_2d": center_2d, "corners": tag.corners
            }

            if role:
                cv2.putText(frame, f"{role.upper()} [{tag_id}]", (center_2d[0]-30, center_2d[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if role == "base":
                self.base_positions.append(tvec)
                self.base_center_2d_list.append(center_2d)

            elif role == "cont":
                self.cont_position = tvec
                self.cont_center_2d = center_2d

        if self.base_positions:
            self.base_position = np.mean(self.base_positions, axis=0)
            self.base_center_2d = tuple(np.mean(self.base_center_2d_list, axis=0).astype(int))
            base_rvecs = [self.detected_tags[tag_id]["rvec"] for tag_id in self.detected_tags if
                          self.detected_tags[tag_id]["role"] == "base"]
            if base_rvecs:
                self.base_rvec = np.mean(base_rvecs, axis=0)
            else:
                self.base_rvec = np.zeros((3, 1), dtype=np.float32)
        else:
            self.base_position = None
            self.base_center_2d = None
            self.base_rvec = np.zeros((3, 1), dtype=np.float32)

        if self.base_position is not None and self.cont_position is not None:
            self.rel_cont_pos = self.cont_position - self.base_position

        # Call draw_tag_boxes to handle all tag visuals
        self.draw_tag_boxes(frame)

        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        # Target is always relative to base
        if self.mode_idx == 0:
            if self.base_rvec is not None:
                # Joystick velocity is in camera frame: [vx, vy, vz]
                joystick_vel = self.velocity.copy()

                # Transform joystick velocity to align with the tag's orientation
                R, _ = cv2.Rodrigues(self.base_rvec)  # Rotation matrix from tag to camera
                tag_aligned_vel = np.dot(R.T, joystick_vel)  # Rotate velocity into the tag's frame

                # Adjust the X and Y components based on the Z-axis contribution
                z_contribution = abs(tag_aligned_vel[2])  # Magnitude of Z-axis component
                max_xy_component = max(abs(tag_aligned_vel[0]), abs(tag_aligned_vel[1]))
                scaling_factor = z_contribution + max_xy_component

                if scaling_factor > 0:
                    tag_aligned_vel[0] *= (scaling_factor / max_xy_component) if max_xy_component > 0 else 1
                    tag_aligned_vel[1] *= (scaling_factor / max_xy_component) if max_xy_component > 0 else 1

                # Apply the joystick velocity in the tag's frame
                self.target_position[0] += tag_aligned_vel[0] * dt  # X-axis (tag frame)
                self.target_position[1] += tag_aligned_vel[1] * dt  # Y-axis (tag frame)
                self.target_position[2] += self.z_velocity * dt  # Apply Z-axis from joystick


                # Clip the target position to stay within valid bounds
                self.target_position = np.clip(self.target_position, -0.5, 0.5)

                # Update the tag label with the new target position
                target = self.target_position.copy()
                self.tag_label.setText(f"Joystick: X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f}")

        elif self.mode_idx == 1:
            if self.rel_cont_pos is not None:
                target = self.rel_cont_pos.copy()
                self.tag_label.setText(f"Hand: X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f}")
            else:
                target = None
                self.tag_label.setText("Hand: Controller or base not detected")
        else:
            target = None

        # Transform target to absolute for visualization
        if self.base_position is not None and self.base_rvec is not None and target is not None:

            # --- Target marker overlay ---
            R, _ = cv2.Rodrigues(self.base_rvec)
            target_abs = np.dot(R, target) + self.base_position
            marker_3d = np.array([target_abs], dtype=np.float32)
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.zeros((3, 1), dtype=np.float32)
            marker_2d, _ = cv2.projectPoints(marker_3d, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            self.target_marker_2d = tuple(marker_2d[0][0].astype(int))
            cv2.drawMarker(frame, self.target_marker_2d, (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(frame, "TARGET", (self.target_marker_2d[0]+10, self.target_marker_2d[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # --- Inverse Kinematics ---
            ik_result = self.compute_ik_3dof(target, link_lengths=(0.093, 0.093, 0.030))
            if ik_result is not None:
                base_angle_deg, shoulder_angle_deg, elbow_angle_deg, wrist_angle_deg = ik_result

                # Forward kinematics (relative to base)
                shoulder_length, elbow_length, wrist_length = 0.093, 0.093, 0.030
                base_angle_rad = np.radians(base_angle_deg)
                shoulder_angle_rad = np.radians(shoulder_angle_deg)
                elbow_angle_rad = np.radians(elbow_angle_deg)
                wrist_angle_rad = np.radians(wrist_angle_deg)

                # Joint positions (relative to base)
                shoulder_3d = np.zeros(3, dtype=np.float32)
                elbow_3d = shoulder_3d + np.array([
                    shoulder_length * np.cos(base_angle_rad) * np.cos(shoulder_angle_rad),
                    shoulder_length * np.sin(base_angle_rad) * np.cos(shoulder_angle_rad),
                    shoulder_length * np.sin(shoulder_angle_rad)
                ])
                wrist_3d = elbow_3d + np.array([
                    elbow_length * np.cos(base_angle_rad) * np.cos(shoulder_angle_rad + elbow_angle_rad),
                    elbow_length * np.sin(base_angle_rad) * np.cos(shoulder_angle_rad + elbow_angle_rad),
                    elbow_length * np.sin(shoulder_angle_rad + elbow_angle_rad)
                ])
                ee_3d = wrist_3d + np.array([
                    wrist_length * np.cos(base_angle_rad) * np.cos(shoulder_angle_rad + elbow_angle_rad + wrist_angle_rad),
                    wrist_length * np.sin(base_angle_rad) * np.cos(shoulder_angle_rad + elbow_angle_rad + wrist_angle_rad),
                    wrist_length * np.sin(shoulder_angle_rad + elbow_angle_rad + wrist_angle_rad)
                ])

                # Transform joints to absolute
                joints_3d = np.array([shoulder_3d, elbow_3d, wrist_3d, ee_3d], dtype=np.float32)
                joints_abs = np.dot(R, joints_3d.T).T + self.base_position
                joints_2d, _ = cv2.projectPoints(joints_abs, np.zeros((3,1)), np.zeros((3,1)), self.CAMERA_MATRIX, self.DIST_COEFFS)
                joints_2d = joints_2d.reshape(-1, 2)

                # Draw segments
                cv2.line(frame, (int(joints_2d[0][0]), int(joints_2d[0][1])), (int(joints_2d[1][0]), int(joints_2d[1][1])), (0,255,0), 3)   # Shoulder
                cv2.line(frame, (int(joints_2d[1][0]), int(joints_2d[1][1])), (int(joints_2d[2][0]), int(joints_2d[2][1])), (255,0,0), 3)   # Elbow
                cv2.line(frame, (int(joints_2d[2][0]), int(joints_2d[2][1])), (int(joints_2d[3][0]), int(joints_2d[3][1])), (0,0,255), 3)   # Wrist

                # Draw pivots
                for pos in joints_2d:
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 8, (255,255,0), -1)



        status = "Connected" if self.serial_link and self.serial_link.is_open else "Disconnected"
        self.status_label.setText(f"Serial: {status}")

        qt_image = self.process_frame_for_display(frame)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

        # IK output (console print, thresholded)
        if target is not None:
            threshold = 0.001
            if (
                self.last_printed_target is None or
                np.linalg.norm(target - self.last_printed_target) > threshold
            ):
                #print("target (relative to base):", target)
                if self.base_position is not None and self.base_rvec is not None:
                    R, _ = cv2.Rodrigues(self.base_rvec)
                    target_abs = np.dot(R, target) + self.base_position
                    #print("target_abs (world):", target_abs)
                ik_result = self.compute_ik_3dof(target, link_lengths=(0.093, 0.093, 0.030))
                if ik_result is not None:
                    print(f"Desired joint angles (deg): {ik_result}")
                else:
                    print("No valid IK solution for this target.")
                self.last_printed_target = target.copy()

    def process_frame_for_display(self, frame):
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        aspect_w, aspect_h = 4, 3
        
        new_w, new_h = get_aspect_scaled_size(label_width, label_height, aspect_w, aspect_h)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.serial_link:
            close_serial(self.serial_link)
        event.accept()

    def resizeEvent(self, event):
        if event:  # Skip the parent call if event is None
            super().resizeEvent(event)
        self.position_overlays()

    def compute_ik_3dof(self, target_position, link_lengths=(0.093, 0.093, 0.030)):
        """
        Calculates joint angles for a 3DOF arm (base rotation, shoulder, elbow) to reach a target position.
        target_position: np.array([x, y, z])
        link_lengths: tuple of (shoulder_length, elbow_length, wrist_length)
        Returns (base_angle_deg, shoulder_angle_deg, elbow_angle_deg, wrist_angle_deg) or None if unreachable.
        """

        # Unpack target coordinates
        target_x, target_y, target_z = target_position
        end_effector_angle_deg = 0  # Desired end effector orientation (can be parameterized)
        shoulder_length, elbow_length, wrist_length = link_lengths

        # Step 1: Calculate wrist position (subtract wrist link from target)
        wrist_x = target_x - wrist_length * np.cos(np.radians(end_effector_angle_deg))
        wrist_y = target_y - wrist_length * np.sin(np.radians(end_effector_angle_deg))
        wrist_z = target_z
        wrist_position = np.array([wrist_x, wrist_y, wrist_z])

        # Step 2: Calculate base rotation angle (in XY plane)
        base_angle_rad = np.arctan2(wrist_y, wrist_x)

        # Step 3: Project wrist position into the arm's plane
        planar_distance = np.sqrt(wrist_x**2 + wrist_y**2)
        vertical_distance = wrist_z

        # Step 4: Use law of cosines to solve for elbow angle
        cos_elbow_angle = (planar_distance**2 + vertical_distance**2 - shoulder_length**2 - elbow_length**2) / (2 * shoulder_length * elbow_length)
        if abs(cos_elbow_angle) > 1:
            print("Target is out of reach for inverse kinematics.")
            return None

        elbow_angle_rad = np.arccos(cos_elbow_angle)

        # Step 5: Solve for shoulder angle using trigonometry
        shoulder_angle_rad = np.arctan2(vertical_distance, planar_distance) - \
            np.arctan2(elbow_length * np.sin(elbow_angle_rad), shoulder_length + elbow_length * np.cos(elbow_angle_rad))

        # Step 6: Calculate wrist angle (if needed for orientation)
        wrist_angle_rad = np.radians(end_effector_angle_deg) - shoulder_angle_rad - elbow_angle_rad

        # Step 7: Convert all angles to degrees
        base_angle_deg = np.degrees(base_angle_rad)
        shoulder_angle_deg = np.degrees(shoulder_angle_rad)
        elbow_angle_deg = np.degrees(elbow_angle_rad)
        wrist_angle_deg = np.degrees(wrist_angle_rad)

        # Constraints
        base_angle_deg = base_angle_deg % 360  # 0-360
        shoulder_angle_deg = np.clip(shoulder_angle_deg, -90, 90)  # up/down only
        elbow_angle_deg = np.clip(elbow_angle_deg, 0, 135)         # up/down only
        wrist_angle_deg = np.clip(wrist_angle_deg, -90, 90)        # up/down only

        return base_angle_deg, shoulder_angle_deg, elbow_angle_deg, wrist_angle_deg
        

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


class ZJoystickWidget(QWidget):
    zPositionChanged = pyqtSignal(float)  # Emits normalized z in [-1, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 120)  # Taller for vertical movement
        self.center = QPoint(self.width() // 2, self.height() // 2)
        self.knob_pos = self.center
        self.radius = self.height() // 2 - 10
        self.active = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw base (semi-transparent)
        painter.setBrush(QColor(220, 220, 220, 120))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        # Draw knob (more opaque)
        painter.setBrush(QColor(100, 100, 255, 180))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.knob_pos, 18, 18)

    def mousePressEvent(self, event):
        if abs(event.pos().y() - self.center.y()) <= self.radius:
            self.active = True
            self.update_knob(event.pos())

    def mouseMoveEvent(self, event):
        if self.active:
            self.update_knob(event.pos())

    def mouseReleaseEvent(self, event):
        self.active = False
        self.knob_pos = self.center
        self.zPositionChanged.emit(0.0)
        self.update()

    def update_knob(self, pos):
        dy = pos.y() - self.center.y()
        dist = abs(dy)
        if dist > self.radius:
            dy = self.radius if dy > 0 else -self.radius
        self.knob_pos = QPoint(self.center.x(), self.center.y() + int(dy))
        norm_z = dy / self.radius  #  Y for UI
        self.zPositionChanged.emit(norm_z)
        self.update()


class OverlayWidget(QWidget):
    def __init__(self, joystick, z_joystick, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        if joystick:
            layout.addWidget(joystick)
            joystick.setStyleSheet("background: transparent;")
            
        if z_joystick:
            layout.addWidget(z_joystick)
            z_joystick.setStyleSheet("background: transparent;")
            
        layout.addStretch()
        self.setLayout(layout)

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