# apriltag_viewer.py

import cv2
import numpy as np
import yaml
import serial
import time
from pupil_apriltags import Detector

# --- Calibration Globals ---
calibrated = False
calibration_z = 0.0


def setup_serial(port='COM7', baud=115200):
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

def send_multi_servo_command(link, angle: int):
    """Send the same angle to all 3 servos."""
    if link and link.is_open:
        for i in range(1, 4):  # Servo IDs 1, 2, 3
            command = f"{i}:{angle}\n"
            link.write(command.encode('utf-8'))
            print(f"[Serial] Sent: {command.strip()}")
            time.sleep(0.01)  # Small delay helps Arduino catch up


def send_tag_cord_command(x: float, y: float, z: float, link):
    if link and link.is_open:
        if (x,y,z) != (0,0,0):
            coords = [x, y, z]
            for i in range(1, 4):  # Servo IDs 1, 2, 3
                value = coords[i - 1]
                # Map value from range (-0.5 to 0.5) to 0–360 degrees
                angle = int((value + 0.5) * 360)
                angle = max(0, min(360, angle))  # Clamp to [0, 360]
                command = f"{i}:{angle}\n"
                link.write(command.encode('utf-8'))
                print(f"[Serial] Sent: {command.strip()}")
                time.sleep(0.01)  # Give Arduino time to process



def main():
    # --- Serial setup ---
    serial_link = setup_serial()

    # --- Load config ---
    with open("AppBase/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    TAG_ID = config["tag_id"]
    TAG_SIZE = config["tag_size"]
    TAG_FAMILY = config["tag_family"]
    CAMERA_ID = config["camera_id"]
    CAMERA_MATRIX = np.array(config["camera_matrix"], dtype=np.float64)
    DIST_COEFFS = np.array(config["distortion_coefficients"], dtype=np.float64)
    
    # --- Setup detector ---
    detector = Detector(families=TAG_FAMILY)

    # --- Start webcam ---
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[Error] Cannot open camera ID {CAMERA_ID}")
        return

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
            send_tag_cord_command(x,y,z, serial_link)
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
            send_multi_servo_command(serial_link, angle)


    cap.release()
    cv2.destroyAllWindows()
    close_serial(serial_link)


if __name__ == "__main__":
    main()
