# apriltag_viewer.py
import platform
import cv2
import numpy as np
import yaml
import serial
import time
import os
from pupil_apriltags import Detector

# --- Calibration Globals ---
calibrated = False
calibration_z = 0.0


def setup_serial(port, baud=115200):
    try:
        link = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"Serial connected on {port}")
        return link
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None


def close_serial(link):
    if link and link.is_open:
        link.close()
        print("Serial port closed.")

def calibrate_z(z):
    """Set the calibration Z value and mark as calibrated."""
    global calibrated, calibration_z
    calibration_z = z
    calibrated = True
    print(f"[Calibration] Calibrated Z set to {z:.2f}")

def send_relative_z(z):
    if not calibrated:
        print("[Warning] Z not calibrated yet.")
        return z

    dz = z - calibration_z
    output = dz
    
    return output

def send_serial_command(link, command, platform_key, serial_delay=0.01):
    """Send a command to Arduino, handling platform-specific encoding."""
    if not link or not link.is_open:
        print("[Serial] Link not open.")
        return
    if platform_key == "windows":
        link.write((command + "\r\n").encode('utf-8'))
    else:
        link.write((command + "\n").encode('utf-8'))
    print(f"[Serial] Sent: {command.strip()}")
    time.sleep(serial_delay)  # Small delay helps Arduino catch up

def send_multi_servo_command(link, angle: int, platform_key="windows"):
    """Send the same angle to all 3 servos."""
    if link and link.is_open:
        for i in range(1, 4):  # Servo IDs 1, 2, 3
            command = f"{i}:{angle}"
            send_serial_command(link, command, platform_key)

def values_changed(new_vals, last_vals, threshold=0.005):
    """Return True if any value changed by more than threshold."""
    return any(abs(a - b) > threshold for a, b in zip(new_vals, last_vals))

def send_tag_cord_command_single_line(x: float, y: float, z: float, link, platform_key="windows", serial_delay=0.01):
    """
    Send all servo angles in a single serial message, e.g. "1:180 2:180 3:180".
    """
    if not link or not link.is_open:
        return
    coords = [x, y, z]
    angles = []
    for value in coords:
        # Clamp input to [-0.5, 0.5]
        value = max(-0.5, min(0.5, value))
        angle = int((value + 0.5) * 360)
        angle = max(0, min(360, angle))  # Clamp to [0, 360]
        angles.append(angle)
    # Build single command string
    command = f"1:{angles[0]} 2:{angles[1]} 3:{angles[2]}"
    send_serial_command(link, command, platform_key, serial_delay)

def load_config(config_path, platform_key):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    platform_config = config.get(platform_key)
    if not platform_config:
        raise KeyError(f"No config for platform: {platform_key}")
    # Check required keys
    required_keys = ["tag_id", "tag_size", "tag_family", "camera_matrix", "distortion_coefficients"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    if "camera_id" not in platform_config or "serial_port" not in platform_config:
        raise KeyError(f"Missing camera_id or serial_port for platform: {platform_key}")
    return config, platform_config

def main():
    # --- Detect platform ---
    system = platform.system().lower()
    if system == "windows":
        platform_key = "windows"
    else:
        platform_key = "linux"

    # --- Load config ---
    config, platform_config = load_config("AppBase/config.yaml", platform_key)

    TAG_ID = config["tag_id"]
    TAG_SIZE = config["tag_size"]
    TAG_FAMILY = config["tag_family"]
    CAMERA_ID = platform_config.get("camera_id")
    SERIAL_PORT = platform_config.get("serial_port")
    CAMERA_MATRIX = np.array(config["camera_matrix"], dtype=np.float64)
    DIST_COEFFS = np.array(config["distortion_coefficients"], dtype=np.float64)
    SERIAL_DELAY = platform_config.get("serial_delay", 0.01)

    # --- Serial setup ---
    serial_link = setup_serial(port=SERIAL_PORT)

    # --- Setup detector ---
    detector = Detector(families=TAG_FAMILY)

    # --- Start webcam ---
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[Error] Cannot open camera ID {CAMERA_ID}")
        return

    last_sent_coords = [None, None, None]  # Track last sent servo positions

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            tag_position = None

            for tag in tags:
                if tag.tag_id != TAG_ID:
                    continue

                for corner in tag.corners:
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

                obj_pts = np.array([
                    [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                    [ TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                    [ TAG_SIZE / 2,  TAG_SIZE / 2, 0],
                    [-TAG_SIZE / 2,  TAG_SIZE / 2, 0]
                ], dtype=np.float32)
                img_pts = np.array(tag.corners, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS)
                if not success:
                    continue

                axis = np.float32([
                    [0, 0, 0],
                    [0.05, 0, 0],     # X axis
                    [0, 0.05, 0],     # Y axis
                    [0, 0, -0.05]     # Z axis
                ])
                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)
                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)
                cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)

                tag_position = tvec.flatten()

            if tag_position is not None:
                x, y, z = tag_position 
                z = send_relative_z(z)
                coords = [x, y, z]
                if last_sent_coords[0] is None or values_changed(coords, last_sent_coords):
                    send_tag_cord_command_single_line(x, y, z, serial_link, platform_key, SERIAL_DELAY)
                    last_sent_coords = coords
                cv2.putText(frame, f"X: {x:.3f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Y: {y:.3f} m", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {z:.3f} m", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
            cv2.imshow("AprilTag Axis & Corners", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('x')]:  # ESC or 'x' to exit
                break

            if key == ord('c') and tag_position is not None:
                _, _, z = tag_position
                calibrate_z(z)

            if ord('0') <= key <= ord('9'):
                angle = (key - ord('0')) * 10
                send_multi_servo_command(serial_link, angle, platform_key)
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        close_serial(serial_link)


if __name__ == "__main__":
    main()
