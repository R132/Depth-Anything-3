# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert DA3-Streaming output to COLMAP sparse reconstruction format.

COLMAP output structure:
    <base_path>/sparse/0/cameras.txt
    <base_path>/sparse/0/images.txt
    <base_path>/sparse/0/points3D.txt

Input path inference:
    If input_path is "xxx/images" or "xxx/image", base_path = "xxx"
    Otherwise, base_path = input_path
"""

import os
import re
import numpy as np
from pathlib import Path


def get_base_path(image_dir):
    """
    Infer the base path from image directory.
    If image_dir ends with 'images' or 'image', return its parent.
    Otherwise return image_dir itself.
    """
    path = Path(image_dir)
    if path.name.lower() in ("images", "image"):
        return str(path.parent)
    return image_dir


def matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    Uses the trace-based method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def read_camera_poses(poses_path):
    """
    Read camera poses from camera_poses.txt.
    Each line is a flattened 4x4 C2W matrix (16 numbers).
    Returns list of 4x4 numpy arrays.
    """
    poses = []
    with open(poses_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = list(map(float, line.split()))
            if len(values) == 16:
                poses.append(np.array(values).reshape(4, 4))
    return poses


def read_intrinsics(intrinsics_path):
    """
    Read camera intrinsics from intrinsic.txt.
    Each line: fx fy cx cy
    Returns list of (fx, fy, cx, cy) tuples.
    """
    intrinsics = []
    with open(intrinsics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = list(map(float, line.split()))
            if len(values) == 4:
                intrinsics.append(tuple(values))
    return intrinsics


def read_ply_pointcloud(ply_path):
    """
    Read point cloud from PLY file (supports both ASCII and binary_little_endian format).
    Returns numpy array of shape (N, 6) with columns [x, y, z, r, g, b].
    """
    with open(ply_path, "rb") as f:
        # Read header
        header_lines = []
        is_binary = False
        num_vertices = 0

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("format"):
                is_binary = "binary" in line
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        if num_vertices == 0:
            return np.empty((0, 6))

        if is_binary:
            # Binary format: float32 x3 + uint8 x3 per vertex = 15 bytes
            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
            data = np.fromfile(f, dtype=vertex_dtype, count=num_vertices)
            points = np.column_stack(
                [
                    data["x"],
                    data["y"],
                    data["z"],
                    data["red"].astype(float),
                    data["green"].astype(float),
                    data["blue"].astype(float),
                ]
            )
        else:
            # ASCII format
            points = []
            for line in f:
                parts = line.decode("ascii").strip().split()
                if len(parts) >= 6:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                    points.append([x, y, z, r, g, b])
            points = np.array(points)

    return points


def write_cameras_txt(cameras_path, intrinsics, width, height):
    """
    Write COLMAP cameras.txt file.

    Format:
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: N
        1 PINHOLE W H fx fy cx cy
    """
    # Assume single camera model (use first frame's intrinsics)
    fx, fy, cx, cy = intrinsics[0]

    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(images_path, poses, image_names):
    """
    Write COLMAP images.txt file.

    COLMAP stores WORLD-TO-CAMERA (W2C) pose in images.txt:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        <empty line for 2D observations>

    Input poses are CAMERA-TO-WORLD (C2W), so we must invert them.
    """
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}, mean observations per image: 0\n")

        for idx, (pose, name) in enumerate(zip(poses, image_names), start=1):
            # Input pose is C2W, COLMAP needs W2C
            w2c = np.linalg.inv(pose)

            # Extract rotation and translation from W2C matrix
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # Convert rotation matrix to quaternion
            qw, qx, qy, qz = matrix_to_quaternion(R)

            # Normalize quaternion
            qnorm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm

            # Write image header line
            f.write(f"{idx} {qw} {qx} {qy} {qz} {T[0]} {T[1]} {T[2]} 1 {name}\n")

            # Write empty 2D observations line
            f.write("\n")


def write_points3D_txt(points3d_path, points):
    """
    Write COLMAP points3D.txt file.

    Format:
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    with open(points3d_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")

        for idx, point in enumerate(points, start=1):
            x, y, z = point[0], point[1], point[2]
            r, g, b = int(point[3]), int(point[4]), int(point[5])
            # ERROR=0, no track data - no trailing space!
            f.write(f"{idx} {x} {y} {z} {r} {g} {b} 0\n")


# =============================================================================
# COLMAP Output Reader — for verification and round-trip checking
# =============================================================================

def read_colmap_cameras(path):
    """
    Read COLMAP cameras.txt file.
    Returns dict: {camera_id: {model, width, height, params}}
    """
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_colmap_images(path):
    """
    Read COLMAP images.txt file.
    Returns dict: {image_id: {qvec, tvec, camera_id, name}}

    Note: COLMAP stores WORLD-TO-CAMERA pose (qvec, tvec).
    To get camera position in world coordinates (C2W translation):
        camera_center = -R^T @ tvec
    """
    images = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                image_id = int(parts[0])
                qw, qx, qy, qz = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                tx, ty, tz = (
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )
                camera_id = int(parts[8])
                name = parts[9] if len(parts) > 9 else ""
                images[image_id] = {
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camera_id,
                    "name": name,
                }
            # Skip the next line (2D observations)
            f.readline()
    return images


def read_colmap_points3D(path):
    """
    Read COLMAP points3D.txt file.
    Returns dict: {point_id: {xyz, rgb, error, track}}
    """
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                error = float(parts[7])
                track = []
                for i in range(8, len(parts) - 1, 2):
                    track.append((int(parts[i]), int(parts[i + 1])))
                points[point_id] = {
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "track": track,
                }
    return points


def qvec_to_rotmat(qvec):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


def verify_colmap_output(sparse_dir):
    """
    Verify COLMAP output files are valid and self-consistent.

    Checks:
    1. All three files exist and can be parsed
    2. Camera IDs referenced by images exist in cameras.txt
    3. Quaternions are normalized
    4. Rotation matrices are orthonormal (det=1)
    5. Camera positions are reasonable (no NaN/Inf)

    Returns True if all checks pass, False otherwise.
    """
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    points3d_path = os.path.join(sparse_dir, "points3D.txt")

    errors = []

    # 1. Read all files
    try:
        cameras = read_colmap_cameras(cameras_path)
        images = read_colmap_images(images_path)
        points3d = read_colmap_points3D(points3d_path)
    except Exception as e:
        print(f"  FAIL: Could not parse COLMAP files: {e}")
        return False

    if not cameras:
        errors.append("No cameras found")
    if not images:
        errors.append("No images found")

    # 2. Check camera IDs
    camera_ids = set(cameras.keys())
    for img_id, img_data in images.items():
        if img_data["camera_id"] not in camera_ids:
            errors.append(
                f"Image {img_id} ({img_data['name']}) references "
                f"non-existent camera_id={img_data['camera_id']}"
            )

    # 3. Check quaternion normalization and rotation validity
    bad_qvecs = 0
    bad_rots = 0
    for img_id, img_data in images.items():
        qvec = img_data["qvec"]
        qnorm = np.linalg.norm(qvec)
        if abs(qnorm - 1.0) > 1e-6:
            bad_qvecs += 1

        R = qvec_to_rotmat(qvec)
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-5:
            bad_rots += 1

    if bad_qvecs > 0:
        errors.append(f"{bad_qvecs} images have non-normalized quaternions (norm != 1.0)")
    if bad_rots > 0:
        errors.append(f"{bad_rots} images have invalid rotation matrices (det(R) != 1)")

    # 4. Check camera positions for NaN/Inf
    nan_positions = 0
    for img_id, img_data in images.items():
        tvec = img_data["tvec"]
        R = qvec_to_rotmat(img_data["qvec"])
        camera_center = -R.T @ tvec  # W2C -> camera position in world coords
        if np.any(np.isnan(camera_center)) or np.any(np.isinf(camera_center)):
            nan_positions += 1

    if nan_positions > 0:
        errors.append(f"{nan_positions} images have NaN/Inf camera positions")

    # 5. Check 3D points
    nan_points = 0
    for pt_id, pt_data in points3d.items():
        if np.any(np.isnan(pt_data["xyz"])) or np.any(np.isinf(pt_data["xyz"])):
            nan_points += 1

    if nan_points > 0:
        errors.append(f"{nan_points} 3D points have NaN/Inf coordinates")

    # Print results
    print(f"  Cameras: {len(cameras)}")
    print(f"  Images:  {len(images)}")
    print(f"  Points:  {len(points3d)}")

    if errors:
        print(f"  Verification FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"    - {err}")
        return False
    else:
        print("  Verification PASSED ✓")
        return True


def output2colmap(output_dir, image_dir):
    """
    Convert DA3-Streaming output to COLMAP sparse reconstruction format.

    Args:
        output_dir: Directory containing DA3-Streaming output files
                    (camera_poses.txt, intrinsic.txt, pcd/)
        image_dir: Original image directory (used to infer base path)

    Returns:
        str: Path to the COLMAP sparse/0 directory
    """
    # Infer base path
    base_path = get_base_path(image_dir)
    sparse_dir = os.path.join(base_path, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # Read input files
    poses_path = os.path.join(output_dir, "camera_poses.txt")
    intrinsics_path = os.path.join(output_dir, "intrinsic.txt")

    poses = read_camera_poses(poses_path)
    intrinsics = read_intrinsics(intrinsics_path)

    if not poses:
        raise ValueError(f"No camera poses found in {poses_path}")
    if not intrinsics:
        raise ValueError(f"No intrinsics found in {intrinsics_path}")

    # Get image names from poses directory or image_dir
    image_names = []
    img_extensions = (".jpg", ".jpeg", ".png")
    for ext in img_extensions:
        image_names.extend(sorted(Path(image_dir).glob(f"*{ext}")))
    image_names = [p.name for p in image_names]

    if len(image_names) != len(poses):
        print(
            f"Warning: Found {len(image_names)} images but {len(poses)} poses. "
            "Using available poses."
        )

    # Determine image dimensions (from first pose count and intrinsics)
    # Use a default or read from first image if available
    width, height = 504, 280  # Default DA3 output resolution
    if image_names:
        first_img = Path(image_dir) / image_names[0]
        if first_img.exists():
            try:
                from PIL import Image
                img = Image.open(first_img)
                width, height = img.size
            except Exception:
                pass

    # Write COLMAP files
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    points3d_path = os.path.join(sparse_dir, "points3D.txt")

    write_cameras_txt(cameras_path, intrinsics, width, height)
    write_images_txt(images_path, poses, image_names[: len(poses)])

    # Write point cloud if available
    pcd_path = os.path.join(output_dir, "pcd", "combined_pcd.ply")
    if os.path.exists(pcd_path):
        points = read_ply_pointcloud(pcd_path)
        write_points3D_txt(points3d_path, points)
        print(f"  Points3D: {len(points)} points written to {points3d_path}")
    else:
        # Create empty points3D.txt
        with open(points3d_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0, mean track length: 0\n")
        print(f"  No point cloud found at {pcd_path}, created empty points3D.txt")

    print(f"COLMAP output written to {sparse_dir}")

    # Verify output
    print("  Verifying COLMAP output...")
    verify_colmap_output(sparse_dir)

    return sparse_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DA3-Streaming output to COLMAP format")
    parser.add_argument("--output_dir", type=str, required=True, help="DA3-Streaming output directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Original image directory")
    args = parser.parse_args()

    output2colmap(args.output_dir, args.image_dir)
