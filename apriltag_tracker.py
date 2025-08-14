import sys
import platform
import cv2
import numpy as np
import yaml
import serial
import time
import os
import threading
import queue
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QSizePolicy, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from pupil_apriltags import Detector


class SerialClient:
    def __init__(self, port, baud, timeout_s=0.1):
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self.ser = None
        self.rx_queue = queue.Queue()
        self.running = False
        self.last_pong = 0.0

    @property
    def connected(self):
        # Consider alive if port open and PONG seen recently (4s window)
        if not (self.ser is not None and self.ser.is_open):
            return False
        return (time.time() - self.last_pong) < 4.0

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()
            return True
        except Exception as e:
            self.ser = None
            return False

    def close(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def send_line(self, line):
        if self.ser and self.ser.is_open:
            try:
                # Arduino reads until '\n'; '\n' is fine on Windows too.
                self.ser.write((line.strip() + "\n").encode("utf-8"))
                return True
            except Exception:
                return False
        return False

    def _reader(self):
        buf = b""
        while self.running and self.ser and self.ser.is_open:
            try:
                chunk = self.ser.read(128)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(errors="replace").strip()
                    if text == "PONG" or text == "READY":
                        self.last_pong = time.time()
                    self.rx_queue.put(text)
            except Exception:
                break


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

    if "serial_port" not in platform_config:
        raise KeyError(f"Missing serial_port for platform: {platform_key}")
    if platform_config["serial_port"] is None or (isinstance(platform_config["serial_port"], str) and platform_config["serial_port"].strip() == ""):
        raise ValueError(f"serial_port for platform {platform_key} is None or empty!")

    return config, platform_config

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def angle_to_pulse(angle_deg, min_us=500, max_us=2500):
    """Map angle [-90, +90] to pulse width [500, 2500] µs."""
    angle = float(np.clip(angle_deg, -90.0, 90.0))
    t = (angle + 90.0) / 180.0  # 0..1
    return int(round(min_us + t * (max_us - min_us)))

def get_aspect_scaled_size(label_width, label_height, aspect_w=4, aspect_h=3):
    # Returns (new_width, new_height) that fits in label and keeps aspect ratio
    if label_width / aspect_w < label_height / aspect_h:
        new_w = label_width
        new_h = int(label_width * aspect_h / aspect_w)
    else:
        new_h = label_height
        new_w = int(label_height * aspect_w / aspect_h)
    return new_w, new_h

def compute_ik_3dof(target_position, link_lengths=(0.093, 0.093, 0.030), wrist_angle_target=0):
    x, y, z = target_position
    L1, L2, L3 = link_lengths

    # 1. Base rotation (theta0), in XY plane
    theta0 = np.arctan2(y, x)

    # 2. Project target into XZ plane
    r = np.sqrt(x**2 + y**2)

    # 3. Adjust for wrist offset
    if wrist_angle_target is None:
        # Point wrist toward the target if no orientation specified
        angle_to_target = np.arctan2(z, r)
        wx = r - L3 * np.cos(angle_to_target)
        wz = z - L3 * np.sin(angle_to_target)
    else:
        wx = r - L3 * np.cos(wrist_angle_target)
        wz = z - L3 * np.sin(wrist_angle_target)

    # 4. Check reachability
    reach = np.sqrt(wx**2 + wz**2)
    if reach > (L1 + L2) or reach < abs(L1 - L2):
        return None  # unreachable

    # 5. Elbow angle (theta2)
    cos_theta2 = (wx**2 + wz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # Elbow-down version (mirror)
    theta2 = -np.arccos(clamp(cos_theta2, -1.0, 1.0))  # NEGATIVE = inverted elbow

    phi = np.arctan2(wz, wx)
    psi = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = phi - psi

    # 7. Wrist angle (theta3) if needed
    if wrist_angle_target is not None:
        theta3 = wrist_angle_target - (theta1 + theta2)
    else:
        theta3 = 0

    # Return angles in degrees
    return (
        np.degrees(theta0),  # Base
        np.degrees(theta1),  # Shoulder
        np.degrees(theta2),  # Elbow
        np.degrees(theta3)   # Wrist (optional, may be used for servo alignment)
    )

class AprilTagTracker(QWidget):
    MODES = ["Joystick", "Hand Tracking"]

    def __init__(self):
        super().__init__()

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
        self.TAG_ROLES = self.config.get("tag_roles", {})  # {id: "base"/"cont"}
        self.CAMERA_MATRIX = np.array(self.config["camera_matrix"], dtype=np.float32).reshape(3, 3)
        self.DIST_COEFFS = np.array(self.config["distortion_coefficients"], dtype=np.float32)

        self.CAMERA_ID = None 
        self.SERIAL_PORT = self.platform_config.get("serial_port")
        self.BAUD_RATE = self.platform_config.get("baud_rate")
        self.SERIAL_DELAY = self.platform_config.get("serial_delay")

        # Serial client + logging
        self.serial_client = SerialClient(self.SERIAL_PORT, self.BAUD_RATE, timeout_s=0.1)
        self.serial_log_path = os.path.join(os.path.dirname(__file__), "serial_log.txt")
        try:
            with open(self.serial_log_path, "w", encoding="utf-8") as lf:
                lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} --- app start ---\n")
        except Exception:
            pass

        # State
        self.detector = Detector(families=self.TAG_FAMILY)

        # Multi-tag state
        self.detected_tags = {}

        self.base_positions = []      # List of np.array for all base tags
        self.base_center_2d_list = [] # List of 2D centers for all base tags

        self.base_cam = None     # Average of base_positions
        self.base_rvec = None

        self.cont_cam = None

        # Target/velocity system
        self.target_rel = np.zeros(3, dtype=np.float32) # initial target 0,0,0 in cam cords
        self.target_camera = np.zeros(3, dtype=np.float32)

        self.last_printed_target = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.z_velocity = 0.0  # For Z-axis joystick control
        self.last_update_time = time.time()
        self.last_sent_us = {"15": None, "14": None, "13": None, "12": None}
        self.last_serial_time = 0.0

        # Mode handling
        self.mode_idx = 0  # 0=Joystick, 1=Hand Tracking
        self.ik_result = None

        self.setup_ui()

    # Map IK angles to servo specific offest angles
    def remap_angles(self, ik_angles_deg):
        try:
            base_raw, shoulder_raw, elbow_raw, wrist_raw = ik_angles_deg
        except Exception:
            return None
        base_deg = base_raw
        shoulder_deg = shoulder_raw - 90  # 0 = +Z
        elbow_deg = elbow_raw + 90       # 0 = L-shape
        wrist_deg = wrist_raw
        return (base_deg, shoulder_deg, elbow_deg, wrist_deg)

    # log serial lines to file
    def _log_serial(self, direction, text):
        # direction: '>' sent, '<' received
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S ')} {direction} {text}\n"
        try:
            with open(self.serial_log_path, "a") as lf:
                lf.write(line)
        except Exception:
            pass
        print(line.strip())

    # send command using SerialClient + optional delay
    def send_serial(self, cmd):
        if not (self.serial_client and self.serial_client.ser and self.serial_client.ser.is_open):
            print("[Serial] Not open")
            return False
        ok = self.serial_client.send_line(cmd)
        if ok:
            self._log_serial(">", cmd)
            try:
                time.sleep(float(self.SERIAL_DELAY) if self.SERIAL_DELAY else 0.0)
            except Exception:
                pass
        else:
            print("[Serial] Write failed")
        return ok

    # Drain RX queue periodically and log also do status update
    def poll_serial(self):
        if not self.serial_client:
            return
        while not self.serial_client.rx_queue.empty():
            text = self.serial_client.rx_queue.get()
            if text:
                self._log_serial("<", text)

        status = "Connected" if self.serial_client.connected else "Disconnected"
        self.status_label.setText(f"Serial: {status}")

    # Use SerialClient to connect/disconnect
    def handle_connect(self):
        if self.serial_client and self.serial_client.ser and self.serial_client.ser.is_open:
            self.serial_client.close()
            self.status_label.setText("Serial: Disconnected")
            self.connect_btn.setText("Connect Serial")
        else:
            ok = self.serial_client.open()
            if ok:
                self.status_label.setText("Serial: Connected")
                self.connect_btn.setText("Disconnect Serial")
            else:
                self.status_label.setText("Serial: Failed to connect")

    def send_servos_from_ik(self, ik_angles_deg):
        adjusted = self.remap_angles(ik_angles_deg)
        if adjusted is None:
            return
        base_deg, shoulder_deg, elbow_deg, wrist_deg = adjusted

        # Ensure angles are valid for your convention
        for val in [base_deg, shoulder_deg, elbow_deg, wrist_deg]:
            if not (-90 <= val <= 90):
                print("IK output out of range, skipping output.")
                return

        # Map each joint to its PCA9685 channel
        ch_angle = [
            ("15", base_deg),      # Base
            ("14", shoulder_deg),  # Shoulder
            ("13", elbow_deg),     # Elbow
            ("12", wrist_deg),     # Wrist
        ]

        # Convert to microseconds and send as: P <ch> <us>
        for ch, ang in ch_angle:
            us = angle_to_pulse(ang, 500, 2500)  # returns int µs
            last = self.last_sent_us.get(ch)
            if last is None or abs(us - last) >= 5: # single us change threshold
                self.send_serial(f"P {int(ch)} {int(us)}")
                self.last_sent_us[ch] = us

    def setup_ui(self):
        """Set up the UI elements and layout."""
        # --- Main window ---
        self.setWindowTitle("AprilTag Tracker - Video")
        self.resize(800, 600)
        main_layout = QVBoxLayout()

        # Video label
        self.video_label = QLabel() # define item
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        main_layout.addWidget(self.video_label) # create item

        # Joystick elements
        self.joystick = JoystickWidget() # define item
        self.z_joystick = ZJoystickWidget() # define item
        self.joystick.positionChanged.connect(self.handle_joystick) # signal
        self.z_joystick.zPositionChanged.connect(self.handle_z_joystick) # signal

        # Overlay widgets
        self.left_overlay = OverlayWidget(self.joystick, None, self.video_label) # define item
        self.left_overlay.setParent(self.video_label)
        self.left_overlay.setFixedSize(120, 120)
        self.left_overlay.show()

        self.right_overlay = OverlayWidget(None, self.z_joystick, self.video_label) # define item
        self.right_overlay.setParent(self.video_label)
        self.right_overlay.setFixedSize(60, 120)
        self.right_overlay.show()

        # --- Control panel window ---
        self.ctrl_win = QWidget() # define item
        self.ctrl_win.setWindowTitle("Control Panel")
        self.ctrl_win.resize(250, 400)
        control_layout = QVBoxLayout()
        
        # Camera selector
        self.camera_select = QComboBox() # define item
        for idx in range(6):
            self.camera_select.addItem(str(idx))
        control_layout.insertWidget(0, QLabel("Camera ID:")) # create item
        control_layout.insertWidget(1, self.camera_select) # create item

        # Open video button
        self.toggle_video_btn = QPushButton("Open Video Window") # define item
        self.toggle_video_btn.clicked.connect(self.toggle_video_window) # signal
        control_layout.insertWidget(2, self.toggle_video_btn) # create item

        # Connect button
        self.connect_btn = QPushButton("Connect Serial") # define item
        self.connect_btn.clicked.connect(self.handle_connect) # signal
        control_layout.addWidget(self.connect_btn) # create item

        # Exit button
        self.exit_btn = QPushButton("Exit") # define item
        self.exit_btn.clicked.connect(self.close) # signal
        control_layout.addWidget(self.exit_btn) # create item

        # Add some spacing
        control_layout.addSpacing(20)

        # Status label
        self.status_label = QLabel("Serial: Disconnected") # define item
        control_layout.addWidget(self.status_label) # create item

        # Tag label
        self.tag_label = QLabel("Tag: Not detected") # define item
        control_layout.addWidget(self.tag_label) # create item

        # Mode toggle button          
        self.mode_toggle_btn = QPushButton(f"Mode: {self.MODES[self.mode_idx]}") # define item
        self.mode_toggle_btn.clicked.connect(self.handle_mode_toggle) # signal
        control_layout.addWidget(self.mode_toggle_btn) # create item

        control_layout.addStretch()
        
        # Set control panel layout and main layout
        self.ctrl_win.setLayout(control_layout)
        self.ctrl_win.show() # Show control panel window on startup
        self.setLayout(main_layout) 

        # Position overlays at startup
        QTimer.singleShot(0, self.position_overlays)

        # Timer for update cycle
        self.timer = QTimer() # define item
        self.timer.timeout.connect(self.update_frame) # signal
        self.timer.start(30) # create item

        # Serial RX poll timer
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.poll_serial)
        self.serial_timer.start(100)

        # Serial PING timer
        self.ping_timer = QTimer()
        self.ping_timer.timeout.connect(lambda: self.send_serial("PING") if (self.serial_client and self.serial_client.ser and self.serial_client.ser.is_open) else None)
        self.ping_timer.start(1500)

    def position_overlays(self):
        # Get video dimensions
        video_w = self.video_label.width()
        video_h = self.video_label.height()

        # Position left overlay (regular joystick) inside the video area on the left side
        left_overlay_h = self.left_overlay.height()
        left_overlay_w = self.left_overlay.width()
        left_x = max(0, 10)  # Small margin from left edge
        left_y = max(0, (video_h - left_overlay_h) // 2)  # Center vertically within the video
        self.left_overlay.move(left_x, left_y)

        # Position right overlay (Z joystick) inside the video area on the right side
        right_overlay_w = self.right_overlay.width()
        right_overlay_h = self.right_overlay.height()
        right_x = max(0, video_w - right_overlay_w - 10)  # Small margin from right edge
        right_y = max(0, (video_h - right_overlay_h) // 2)  # Center vertically within the video
        self.right_overlay.move(right_x, right_y)

    def open_video_window(self, cam_id=None):
        """Open (or re-open) the video window on the selected camera."""
        if cam_id is None:
            cam_id = int(self.camera_select.currentText())
        
        print(f"Opening camera: {cam_id}")
        
        # Release old capture
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

        # Try to open the camera
        self.cap = cv2.VideoCapture(cam_id)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self.ctrl_win, "Camera Error", f"Cannot open camera #{cam_id}")
            self.cap = None  # Ensure cap is None if opening failed
            return False
        
        self.show()  # Show the video window
        self.CAMERA_ID = cam_id
        self.toggle_video_btn.setText("Close Video Window")
        return True

    def toggle_video_window(self):
        """Toggle video window on/off."""
        if self.isVisible():
            # Close the video window but leave the control panel running
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
                self.cap = None
            self.hide()
            self.toggle_video_btn.setText("Open Video Window")
        else:
            # Try to open the video window
            success = self.open_video_window()
            if not success:
                # If opening failed, keep the button text as "Open Video Window"
                self.toggle_video_btn.setText("Open Video Window")

    def closeEvent(self, event):
        # Clean up camera resource
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if self.serial_client:
            self.serial_client.close()
        # Close control panel too
        if hasattr(self, 'ctrl_win'):
            self.ctrl_win.close()
        event.accept()

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
        self.z_velocity = z * -0.05  # Adjust scaling factor as needed

    def handle_input(self, dt):
        
        if self.base_cam is not None and self.target_rel is not None: # Check if base is detected

            if self.mode_idx == 0:  # Joystick mode

                # Joystick velocity is in camera frame: [vx, vy, vz]
                joystick_vel = self.velocity.copy()

                # Transform joystick velocity to align with the tag's orientation
                R, _ = cv2.Rodrigues(self.base_rvec)  # Convert rotation matrix to rotation vector
                tag_aligned_vel = np.dot(R.T, joystick_vel)  # Rotate velocity to align with tag's frame

                # Apply the joystick velocity in the tag's frame
                self.target_rel[0] += tag_aligned_vel[0] * dt  # X-axis (tag frame)
                self.target_rel[1] += tag_aligned_vel[1] * dt  # Y-axis (tag frame)
                self.target_rel[2] += self.z_velocity * dt  # Apply Z-axis from joystick

                # Clip the target position to stay within valid bounds
                self.target_rel = np.clip(self.target_rel, -0.5, 0.5)

                # Update the tag label with the new target position
                self.tag_label.setText(f"Target_relative: X={self.target_rel[0]:.3f} Y={self.target_rel[1]:.3f} Z={self.target_rel[2]:.3f}")

                

            elif self.mode_idx == 1: # Hand Tracking mode

                if self.cont_cam is not None: # Check if controller tag is detected

                    # Use the controller's position as the target position
                    self.target_rel = self.cont_cam.copy()

                    # Update the tag label with the new target position
                    self.tag_label.setText(f"Controler_camera: X={self.target_rel[0]:.3f} Y={self.target_rel[1]:.3f} Z={self.target_rel[2]:.3f}")
                else:
                    self.tag_label.setText("Controller not detected")
            
            else:
                self.tag_label.setText("Unknown mode selected")

        else:
            self.tag_label.setText("Base not detected")

    def detect_tags(self, frame):
        tags = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        self.detected_tags.clear()
        self.base_positions.clear()
        self.base_cam = None
        self.base_rvec = None

        for tag in tags:
            tag_id = tag.tag_id
            role = self.TAG_ROLES.get(str(tag_id), None)

            obj_pts = np.array([
                [-self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0],  # Bottom-left
                [self.TAG_SIZE / 2, -self.TAG_SIZE / 2, 0],   # Bottom-right  
                [self.TAG_SIZE / 2, self.TAG_SIZE / 2, 0],    # Top-right
                [-self.TAG_SIZE / 2, self.TAG_SIZE / 2, 0]    # Top-left
            ], dtype=np.float32)

            img_pts = np.array(tag.corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.CAMERA_MATRIX, self.DIST_COEFFS)

            if not success or np.linalg.norm(tvec) > 10:
                continue

            tvec = tvec.flatten()
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            center_2d = tuple(center_2d[0][0].astype(int))

            self.detected_tags[tag_id] = {
                "role": role, "tvec": tvec, "rvec": rvec, 
                "center_2d": center_2d, "corners": tag.corners
            }

            if role == "base":
                self.base_positions.append(tvec)

        if self.base_positions:
            self.base_cam = np.mean(self.base_positions, axis=0)
            base_rvecs = [self.detected_tags[tag_id]["rvec"] for tag_id in self.detected_tags 
                         if self.detected_tags[tag_id]["role"] == "base"]
            if base_rvecs:
                self.base_rvec = np.mean(base_rvecs, axis=0)

    def draw_tags(self, frame):
        for tag_id, tag_data in self.detected_tags.items():
            center_2d = tag_data["center_2d"]
            role = tag_data["role"]

            # Draw tag label
            cv2.putText(frame, f"{role.upper() if role else 'UNKNOWN'} [{tag_id}]", 
                       (center_2d[0] - 30, center_2d[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw tag corners
            for corner in tag_data["corners"]:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

            # Draw tag axes
            rvec, tvec = tag_data["rvec"], tag_data["tvec"]
            axis = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.CAMERA_MATRIX, self.DIST_COEFFS)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)  # X-axis (red)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)  # Y-axis (green)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)  # Z-axis (blue)

    def draw_target(self, frame):
        if self.base_cam is None or self.base_rvec is None:
            return

        # Transform target to camera coordinates
        R, _ = cv2.Rodrigues(self.base_rvec)
        self.target_camera = np.dot(R, self.target_rel) + self.base_cam
         
         # Project to 2D
        marker_3d = np.array([self.target_camera], dtype=np.float32)
        marker_2d, _ = cv2.projectPoints(marker_3d, np.zeros((3, 1)), np.zeros((3, 1)), 
                                            self.CAMERA_MATRIX, self.DIST_COEFFS)
        target_2d = tuple(marker_2d[0][0].astype(int))

        # Draw target marker
        cv2.drawMarker(frame, target_2d, (0, 0, 255), markerType=cv2.MARKER_CROSS, 
            markerSize=20, thickness=3)
        cv2.putText(frame, "TARGET", (target_2d[0] + 15, target_2d[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def handle_ik(self, frame):
        if self.target_camera is not None and self.base_cam is not None:
            threshold = 0.001 # Threshold for IK calculations

            if (self.last_printed_target is None or np.linalg.norm(self.target_rel - self.last_printed_target) > threshold):
                self.ik_result = compute_ik_3dof(self.target_rel, link_lengths=(0.093, 0.093, 0.030))
                
                # --- Add this block ---
                valid = False
                if self.ik_result is not None:
                    valid = True
                #if self.ik_result is not None:
                    # Check all joint angles are within -90 to +90 degrees
                    #if all(-90.0 <= angle <= 90.0 for angle in self.ik_result):
                        #valid = True
                   # else:
                        #print("IK output out of range, skipping output.")
                        #self.ik_result = None  # Mark as invalid
                # --- End block ---

                if valid:
                    #print(f"UnAdjusted Joint angles: {self.ik_result[0]:.2f}, {self.ik_result[1]:.2f}, {self.ik_result[2]:.2f}, {self.ik_result[3]:.2f}")
                    self.send_servos_from_ik(self.ik_result)
                else:
                    print("target unreachable, IK failed or out of range.")

                self.last_printed_target = self.target_rel.copy()
            self.draw_arm(frame, self.ik_result)

    def draw_arm(self, frame, ik_result):
        if ik_result is None or self.base_cam is None or self.base_rvec is None:
            return

        # Use raw IK angles (deg -> rad) for geometry
        theta0, theta1, theta2, theta3 = np.radians(ik_result)
        L1, L2, L3 = 0.093, 0.093, 0.030

        # Also compute adjusted angles (deg) for labeling only
        adjusted = self.remap_angles(ik_result)  # degrees
        if adjusted is None:
            return
        base_adj_deg, shoulder_adj_deg, elbow_adj_deg, wrist_adj_deg = adjusted

        # Base frame origin
        base_3d = np.array([0, 0, 0], dtype=np.float32)

        # Direction vector in XY plane (base rotation)
        dx = np.cos(theta0)
        dy = np.sin(theta0)

        # Shoulder joint position (just base here)
        shoulder_3d = base_3d

        # Elbow position
        elbow_3d = shoulder_3d + np.array([
            L1 * dx * np.cos(theta1),
            L1 * dy * np.cos(theta1),
            L1 * np.sin(theta1)
        ], dtype=np.float32)

        # Wrist position
        elbow_angle_total = theta1 + theta2
        wrist_3d = elbow_3d + np.array([
            L2 * dx * np.cos(elbow_angle_total),
            L2 * dy * np.cos(elbow_angle_total),
            L2 * np.sin(elbow_angle_total)
        ], dtype=np.float32)

        # End effector position
        wrist_angle_total = elbow_angle_total + theta3
        end_effector_3d = wrist_3d + np.array([
            L3 * dx * np.cos(wrist_angle_total),
            L3 * dy * np.cos(wrist_angle_total),
            L3 * np.sin(wrist_angle_total)
        ], dtype=np.float32)

        # Transform to camera coordinates
        R, _ = cv2.Rodrigues(self.base_rvec)
        joints_3d_cam = np.array([
            np.dot(R, shoulder_3d) + self.base_cam,
            np.dot(R, elbow_3d) + self.base_cam,
            np.dot(R, wrist_3d) + self.base_cam,
            np.dot(R, end_effector_3d) + self.base_cam
        ], dtype=np.float32)

        # Project to 2D
        joints_2d, _ = cv2.projectPoints(
            joints_3d_cam,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.CAMERA_MATRIX,
            self.DIST_COEFFS
        )
        joints_2d = joints_2d.reshape(-1, 2)

        if len(joints_2d) < 4:
            return

        # Draw arm segments
        cv2.line(frame, tuple(joints_2d[0].astype(int)), tuple(joints_2d[1].astype(int)), (0, 255, 0), 4) # bicep
        cv2.line(frame, tuple(joints_2d[1].astype(int)), tuple(joints_2d[2].astype(int)), (255, 0, 0), 4) # forearm
        cv2.line(frame, tuple(joints_2d[2].astype(int)), tuple(joints_2d[3].astype(int)), (0, 0, 255), 4) # end effector

        # Draw angle sectors; use raw angles for arcs, adjusted angles for labels
        self.draw_angle_sector(frame, joints_2d[0], joints_2d[1], None, theta0, "Base",
                               np.degrees(theta0), (255, 255, 0), label_override_deg=base_adj_deg)
        self.draw_angle_sector(frame, joints_2d[0], joints_2d[0], joints_2d[1], theta1, "Shoulder",
                               np.degrees(theta1), (255, 100, 0), label_override_deg=shoulder_adj_deg)
        self.draw_angle_sector(frame, joints_2d[1], joints_2d[0], joints_2d[2], theta2, "Elbow",
                               np.degrees(theta2), (0, 255, 255), label_override_deg=elbow_adj_deg)
        self.draw_angle_sector(frame, joints_2d[2], joints_2d[2], joints_2d[3], theta3, "wrist",
                               np.degrees(theta3), (0, 255, 255), label_override_deg=wrist_adj_deg)

        # Draw joints
        joint_colors = [(255, 255, 0), (255, 100, 0), (0, 255, 255), (255, 0, 255)]
        for pos, color in zip(joints_2d, joint_colors):
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)

    def draw_angle_sector(self, frame, joint_pos, ref_pos1, ref_pos2, angle_rad, joint_name, angle_deg, color, label_override_deg=None):
        """Draw angle sector and label for a joint.
           angle_rad is used for drawing the arc; label_override_deg (if given) is shown as the label value."""
        joint_x, joint_y = int(joint_pos[0]), int(joint_pos[1])
        
        # Skip if joint is off-screen
        if joint_x < 0 or joint_y < 0 or joint_x >= frame.shape[1] or joint_y >= frame.shape[0]:
            return
            
        radius = 30
        
        if joint_name in ["Base", "Shoulder"]:
            if joint_name == "Base":
                # Base joint (rotation around Z-axis)
                # For base joint, show rotation from reference direction
                start_angle = 0
                end_angle = angle_rad
            elif joint_name == "Shoulder":
                # Shoulder joint - angle between XY plane and arm segment
                # Calculate the horizontal reference direction (XY plane projection)
                # This represents 0 degrees (horizontal)
                start_angle = 0  # Horizontal reference
                
                # Calculate the angle to the actual arm segment (to elbow joint)
                if ref_pos2 is not None:  # ref_pos2 should be joints_2d[1] (elbow position)
                    # Vector from shoulder to elbow in 2D screen coordinates
                    arm_vector = ref_pos2 - joint_pos
                    # Calculate angle of this vector relative to horizontal
                    end_angle = np.arctan2(arm_vector[1], arm_vector[0])
                else:
                    end_angle = angle_rad
        else:
            # For other joints, calculate angle between two vectors
            vec1 = ref_pos1 - joint_pos
            vec2 = ref_pos2 - joint_pos
            
            # Calculate angles
            angle1 = np.arctan2(vec1[1], vec1[0])
            angle2 = np.arctan2(vec2[1], vec2[0])
            
            # Ensure we draw the smaller arc
            start_angle = angle1
            end_angle = angle2
            if abs(end_angle - start_angle) > np.pi:
                if end_angle > start_angle:
                    start_angle += 2 * np.pi
                else:
                    end_angle += 2 * np.pi

        # Convert to degrees for OpenCV (OpenCV uses degrees)
        start_deg = np.degrees(start_angle)
        end_deg = np.degrees(end_angle)
        
        # Draw the angle arc/sector
        if abs(end_deg - start_deg) > 5:  # Only draw if angle is significant
            # Create a filled sector (pie slice)
            pts = []
            pts.append((joint_x, joint_y))  # Center point
            
            # Add arc points
            angle_range = np.linspace(start_angle, end_angle, 20)
            for a in angle_range:
                x = int(joint_x + radius * np.cos(a))
                y = int(joint_y + radius * np.sin(a))
                pts.append((x, y))
            
            pts.append((joint_x, joint_y))  # Back to center
            
            # Draw filled sector with transparency effect
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw arc outline
            cv2.ellipse(frame, (joint_x, joint_y), (radius, radius), 0, start_deg, end_deg, color, 2)

        # Choose label angle: adjusted (override) if provided, else raw
        label_angle = label_override_deg if label_override_deg is not None else angle_deg
        label_text = f"{joint_name}: {label_angle:.1f}"
        
        # Draw angle label with different positioning for Base and Shoulder
        # Position label offset from joint - stack Base and Shoulder labels
        if joint_name == "Base":
            # Position Base label higher
            label_x = joint_x + 40
            label_y = joint_y - 30  # Higher position for Base
        elif joint_name == "Shoulder":
            # Position Shoulder label lower
            label_x = joint_x + 40
            label_y = joint_y - 10  # Lower position for Shoulder
        else:
            # Default positioning for other joints
            label_x = joint_x + 40
            label_y = joint_y - 10
        
        # Ensure label stays on screen
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        if label_x + text_size[0] > frame.shape[1]:
            label_x = joint_x - text_size[0] - 10
        if label_y < 20:
            if joint_name == "Base":
                label_y = joint_y + 50  # Move Base label further down if too high
            else:
                label_y = joint_y + 30  # Move other labels down
            
        # Draw text background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x - 2, label_y - 15), (label_x + text_size[0] + 2, label_y + 5), (0, 0, 0), -1)
        alpha = 0.5  # Transparency factor (0=fully transparent, 1=opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text
        cv2.putText(frame, label_text, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def update_frame(self):
        # Check if camera is available and working
        if not hasattr(self, 'cap') or self.cap is None:
            return  # Exit early if no camera is set up
        
        if not self.cap.isOpened():
            print("Camera not opened, skipping frame update")
            return  # Exit early if camera is not opened
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return  # Exit early if frame reading failed

        # Detect tags
        self.detect_tags(frame)
        
        # Draw tags
        self.draw_tags(frame)
        
        # Draw target
        self.draw_target(frame)

        # Handle IK
        self.handle_ik(frame)

        # Input integration
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        self.handle_input(dt)

        # Serial status (from SerialClient)
        status = "Connected" if (self.serial_client and self.serial_client.connected) else "Disconnected"
        self.status_label.setText(f"Serial: {status}")

        # Update overlay positions
        self.position_overlays()

        qt_image = self.process_frame_for_display(frame)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def process_frame_for_display(self, frame):
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        aspect_w, aspect_h = 4, 3
        
        new_w, new_h = get_aspect_scaled_size(label_width, label_height, aspect_w, aspect_h)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # Note: Could do this to handle all the video stuff like the scale and aspect ratio
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


class JoystickWidget(QWidget):
    positionChanged = pyqtSignal(float, float)  # Emits normalized x, y in [-1, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self.center = QPoint(self.width() // 2, self.height() // 2)
        self.knob_pos = self.center
        self.radius = min(self.width(), self.height()) // 2 - 10
        self.active = False
        self.touch_id = None  # Track the touch point ID
        self.setAttribute(Qt.WA_AcceptTouchEvents)  # Enable touch events

    def event(self, event):
        # Handle touch events
        if event.type() == QEvent.TouchBegin:
            if not self.touch_id:  # Only accept if we're not tracking a touch
                touch_point = event.touchPoints()[0]
                pos = touch_point.pos().toPoint()
                if (pos - self.center).manhattanLength() <= self.radius:
                    self.touch_id = touch_point.id()
                    self.active = True
                    self.update_knob(pos)
                    return True
        elif event.type() == QEvent.TouchUpdate:
            for touch_point in event.touchPoints():
                if touch_point.id() == self.touch_id:
                    self.update_knob(touch_point.pos().toPoint())
                    return True
        elif event.type() == QEvent.TouchEnd:
            for touch_point in event.touchPoints():
                if touch_point.id() == self.touch_id:
                    self.active = False
                    self.touch_id = None
                    self.knob_pos = self.center
                    self.positionChanged.emit(0.0, 0.0)
                    self.update()
                    return True
        return super().event(event)

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
        self.touch_id = None  # Track the touch point ID
        self.setAttribute(Qt.WA_AcceptTouchEvents)  # Enable touch events

    def event(self, event):
        # Handle touch events
        if event.type() == QEvent.TouchBegin:
            if not self.touch_id:  # Only accept if we're not tracking a touch
                touch_point = event.touchPoints()[0]
                pos = touch_point.pos().toPoint()
                if abs(pos.y() - self.center.y()) <= self.radius:
                    self.touch_id = touch_point.id()
                    self.active = True
                    self.update_knob(pos)
                    return True
        elif event.type() == QEvent.TouchUpdate:
            for touch_point in event.touchPoints():
                if touch_point.id() == self.touch_id:
                    self.update_knob(touch_point.pos().toPoint())
                    return True
        elif event.type() == QEvent.TouchEnd:
            for touch_point in event.touchPoints():
                if touch_point.id() == self.touch_id:
                    self.active = False
                    self.touch_id = None
                    self.knob_pos = self.center
                    self.zPositionChanged.emit(0.0)
                    self.update()
                    return True
        return super().event(event)

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
        self.setAttribute(Qt.WA_AcceptTouchEvents)
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

    def event(self, event):
        # Let touch events propagate to children
        if event.type() in (QEvent.TouchBegin, QEvent.TouchUpdate, QEvent.TouchEnd):
            return False  # Don't handle here, let it pass to children
        return super().event(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents)
    app.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents)
    window = AprilTagTracker()
    sys.exit(app.exec_())