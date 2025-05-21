# apriltag_viewer.py

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector


def main():
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            if tag.tag_id != TAG_ID:
                continue

            # Draw corner points
            for corner in tag.corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

            # Estimate pose
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

            # Draw axes (X=Red, Y=Green, Z=Blue)
            axis = np.float32([
                [0, 0, 0],
                [0.05, 0, 0],     # X axis
                [0, 0.05, 0],     # Y axis
                [0, 0, -0.05]     # Z axis
            ])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)  # X - red
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)  # Y - green
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)  # Z - blue

        cv2.imshow("AprilTag Axis & Corners", frame)
        if cv2.waitKey(1) in [27, ord('x')]:  # ESC or 'x' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
