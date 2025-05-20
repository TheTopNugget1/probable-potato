# apriltag_3d_viewer.py

import cv2
import numpy as np
import trimesh
import pyrender
import yaml
from pupil_apriltags import Detector
import serial
import time


def setup_serial(port='COM7', baud=112500):
    try:
        link = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Give time for the connection to establish
        print(f"Serial connected on {port}")
        return link
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None


def close_serial(link):
    if link and link.is_open:
        link.close()
        print("Serial port closed.")


def send_to_robot(link, command: str):
    if link and link.is_open:
        link.write((command + '\n').encode('utf-8'))


def handle_keys(key, offset):
    step = 0.01
    if key == ord('a'):
        offset[0] -= step
    elif key == ord('d'):
        offset[0] += step
    elif key == ord('w'):
        offset[1] += step
    elif key == ord('s'):
        offset[1] -= step
    elif key == ord('q'):
        offset[2] -= step
    elif key == ord('e'):
        offset[2] += step


def main():
    link = setup_serial()

    # --- Load config ---
    with open("AppBase/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    MODEL_PATH = config["model_path"]
    SCALE = config["model_scale"]
    TAG_ID = config["tag_id"]
    TAG_SIZE = config["tag_size"]
    TAG_FAMILY = config["tag_family"]
    CAMERA_ID = config["camera_id"]
    CAMERA_MATRIX = np.array(config["camera_matrix"], dtype=np.float64)
    DIST_COEFFS = np.array(config["distortion_coefficients"], dtype=np.float64)
    TRANS_OFFS = np.array([
        config["translation_offset"]["x"],
        config["translation_offset"]["y"],
        config["translation_offset"]["z"]
    ], dtype=np.float64)

    # --- Detector & Model ---
    DETECTOR = Detector(families=TAG_FAMILY)
    loaded = trimesh.load(MODEL_PATH, process=False)
    mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()]) if isinstance(loaded, trimesh.Scene) else loaded
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # --- Renderer & Scene ---
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=np.eye(4))
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=3.0), pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(640, 480)
    camera = pyrender.IntrinsicsCamera(
        fx=CAMERA_MATRIX[0, 0], fy=CAMERA_MATRIX[1, 1],
        cx=CAMERA_MATRIX[0, 2], cy=CAMERA_MATRIX[1, 2]
    )
    scene.add(camera, pose=np.eye(4))

    cap = cv2.VideoCapture(CAMERA_ID) # Change to your camera index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = DETECTOR.detect(gray)

        for tag in tags:
            if tag.tag_id != TAG_ID:
                continue

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

            axis = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)

            R, _ = cv2.Rodrigues(rvec)
            R[1, :] *= -1
            R[2, :] *= -1
            t = tvec.flatten()
            t[1] *= -1
            t[2] *= -1

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            scale_mat = np.eye(4)
            scale_mat[:3, :3] *= SCALE
            pose = pose @ scale_mat

            pose[:3, 3] += TRANS_OFFS

            node = scene.add(render_mesh, pose=pose)
            color, _ = renderer.render(scene)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            scene.remove_node(node)

            frame = cv2.addWeighted(frame, 0.5, color, 0.5, 0)

            # --- Placeholder: Send coordinates to robot ---
            tx, ty, tz = pose[:3, 3]
            print(link, f"POS {tx:.3f},{ty:.3f},{tz:.3f}")
            send_to_robot(link, f"POS {tx:.3f},{ty:.3f},{tz:.3f}")

        cv2.imshow("AprilTag Viewer", frame)
        key = cv2.waitKey(1)
        if key in [27, ord('x')]:
            break
        handle_keys(key, TRANS_OFFS)

    cap.release()
    cv2.destroyAllWindows()
    close_serial(link)


if __name__ == "__main__":
    main()
