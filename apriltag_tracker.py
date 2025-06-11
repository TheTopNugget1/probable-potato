import sys
import platform
import cv2
import numpy as np
import yaml
import serial
import time
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
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

class AprilTagTracker(QWidget):
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
        self.TAG_ID = self.config["tag_id"]
        self.TAG_SIZE = self.config["tag_size"]
        self.TAG_FAMILY = self.config["tag_family"]
        self.CAMERA_ID = self.platform_config.get("camera_id")
        self.SERIAL_PORT = self.platform_config.get("serial_port")
        self.CAMERA_MATRIX = np.array(self.config["camera_matrix"], dtype=np.float64)
        self.DIST_COEFFS = np.array(self.config["distortion_coefficients"], dtype=np.float64)
        self.SERIAL_DELAY = self.platform_config.get("serial_delay", 0.01)

        # State
        self.serial_link = None
        self.detector = Detector(families=self.TAG_FAMILY)
        self.cap = cv2.VideoCapture(self.CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera failed to open with ID: {self.CAMERA_ID}")
        self.servo_offsets = [0.0, 0.0, 0.0]
        self.last_sent_coords = [None, None, None]
        self.tag_position = None
        self.calibrated = False
        self.calibration_z = 0.0

        # UI Elements
        self.video_label = QLabel()
        self.status_label = QLabel("Serial: Disconnected")
        self.offset_label = QLabel("Offsets: X=0.00 Y=0.00 Z=0.00")
        self.tag_label = QLabel("Tag: Not detected")
        self.connect_btn = QPushButton("Connect Serial")
        self.calib_btn = QPushButton("Calibrate Z")
        self.reset_btn = QPushButton("Reset Offsets")
        self.exit_btn = QPushButton("Exit")

        # Sliders for offsets
        self.x_slider = QSlider(Qt.Horizontal)
        self.y_slider = QSlider(Qt.Horizontal)
        self.z_slider = QSlider(Qt.Horizontal)
        for slider in (self.x_slider, self.y_slider, self.z_slider):
            slider.setMinimum(-50)
            slider.setMaximum(50)
            slider.setValue(0)
            slider.setTickInterval(1)
            slider.setSingleStep(1)

        # Layout
        controls = QGridLayout()
        controls.addWidget(QLabel("X Offset"), 0, 0)
        controls.addWidget(self.x_slider, 0, 1)
        controls.addWidget(QLabel("Y Offset"), 1, 0)
        controls.addWidget(self.y_slider, 1, 1)
        controls.addWidget(QLabel("Z Offset"), 2, 0)
        controls.addWidget(self.z_slider, 2, 1)
        controls.addWidget(self.connect_btn, 3, 0)
        controls.addWidget(self.calib_btn, 3, 1)
        controls.addWidget(self.reset_btn, 4, 0)
        controls.addWidget(self.exit_btn, 4, 1)
        controls.addWidget(self.status_label, 5, 0, 1, 2)
        controls.addWidget(self.offset_label, 6, 0, 1, 2)
        controls.addWidget(self.tag_label, 7, 0, 1, 2)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, 3)
        main_layout.addLayout(controls, 1)
        self.setLayout(main_layout)

        # Signals
        self.connect_btn.clicked.connect(self.handle_connect)
        self.calib_btn.clicked.connect(self.handle_calibrate)
        self.reset_btn.clicked.connect(self.handle_reset)
        self.exit_btn.clicked.connect(self.close)
        self.x_slider.valueChanged.connect(self.update_offsets)
        self.y_slider.valueChanged.connect(self.update_offsets)
        self.z_slider.valueChanged.connect(self.update_offsets)

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

    def handle_reset(self):
        self.servo_offsets = [0.0, 0.0, 0.0]
        self.x_slider.setValue(0)
        self.y_slider.setValue(0)
        self.z_slider.setValue(0)
        self.offset_label.setText("Offsets: X=0.00 Y=0.00 Z=0.00")

    def update_offsets(self):
        self.servo_offsets[0] = self.x_slider.value() / 100.0
        self.servo_offsets[1] = self.y_slider.value() / 100.0
        self.servo_offsets[2] = self.z_slider.value() / 100.0
        self.offset_label.setText(
            f"Offsets: X={self.servo_offsets[0]:.2f} Y={self.servo_offsets[1]:.2f} Z={self.servo_offsets[2]:.2f}"
        )

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
        tag_position = None

        for tag in tags:
            if tag.tag_id != self.TAG_ID:
                continue

            for corner in tag.corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

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

            tag_position = tvec.flatten()
            # Tag detection logic safety
            if np.linalg.norm(tag_position) > 10:
                print("[Warning] Tag position seems invalid:", tag_position)
                tag_position = None

        self.tag_position = tag_position

        # Live data display
        if tag_position is not None:
            x, y, z = tag_position
            x += self.servo_offsets[0]
            y += self.servo_offsets[1]
            z += self.servo_offsets[2]
            z_rel = z - self.calibration_z if self.calibrated else z
            coords = [x, y, z_rel]
            self.tag_label.setText(f"Tag: X={x:.3f} Y={y:.3f} Z={z_rel:.3f}")
            if self.last_sent_coords[0] is None or any(abs(a - b) > 0.005 for a, b in zip(coords, self.last_sent_coords)):
                send_tag_cord_command_single_line(x, y, z_rel, self.serial_link, self.platform_key, self.SERIAL_DELAY)
                self.last_sent_coords = coords
        else:
            self.tag_label.setText("Tag: Not detected")

        # Show serial status
        status = "Connected" if self.serial_link and self.serial_link.is_open else "Disconnected"
        self.status_label.setText(f"Serial: {status}")

        # Draw overlays
        frame = cv2.resize(frame, (640, 480))
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.serial_link:
            close_serial(self.serial_link)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AprilTagTracker()
    window.show()
    sys.exit(app.exec_())
