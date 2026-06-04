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
Main pipeline: DA3-Streaming + COLMAP sparse reconstruction + scale alignment.

Workflow:
1. Run DA3-Streaming → produces camera poses & COLMAP output at <base>/da3_sparse/0/
   (DA3 has correct scale but sparse geometry)
2. Run COLMAP sparse reconstruction → produces output at <base>/sparse/0/
   (COLMAP has accurate geometry but wrong/unknown scale)
3. Compute scale factor between DA3 and COLMAP camera positions
4. Rename COLMAP output <base>/sparse/0/ → <base>/colmap_sparse/0/
5. Write scale-adjusted COLMAP output to <base>/sparse/0/
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)
from output2colmap import (
    get_base_path,
    output2colmap,
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_points3D,
)




def get_camera_centers(images_dict):
    """
    Extract camera centers in world coordinates from COLMAP images dict.
    COLMAP stores W2C pose: center = -R^T @ tvec
    Returns dict: {image_name: np.array([x, y, z])}
    """
    centers = {}
    for img_id, img_data in images_dict.items():
        R = qvec2rotmat(img_data.qvec)
        tvec = img_data.tvec
        center = -R.T @ tvec
        centers[img_data.name] = center
    return centers


def compute_scale_factor(da3_dir, colmap_dir):
    """
    Compute scale factor between DA3 and COLMAP reconstructions.

    Scale = median(||colmap_center_i - colmap_center_j|| / ||da3_center_i - da3_center_j||)
    over all image pairs.

    Uses median of pairwise distance ratios for robustness.
    """
    da3_images = read_colmap_images(os.path.join(da3_dir, "images.txt"))
    colmap_images = read_colmap_images(os.path.join(colmap_dir, "images.txt"))

    # Find common images by name
    da3_names = set(img.name for img in da3_images.values())
    colmap_names = set(img.name for img in colmap_images.values())
    common_names = sorted(da3_names & colmap_names)

    if len(common_names) < 2:
        raise ValueError(
            f"Need at least 2 common images to compute scale. "
            f"DA3: {len(da3_images)} images, COLMAP: {len(colmap_images)} images, "
            f"Common: {len(common_names)}"
        )

    # Get camera centers for common images
    da3_centers = {}
    colmap_centers = {}
    for img_id, img_data in da3_images.items():
        if img_data.name in common_names:
            R = qvec2rotmat(img_data.qvec)
            da3_centers[img_data.name] = -R.T @ img_data.tvec

    for img_id, img_data in colmap_images.items():
        if img_data.name in common_names:
            R = qvec2rotmat(img_data.qvec)
            colmap_centers[img_data.name] = -R.T @ img_data.tvec

    names = sorted(common_names)

    # Compute pairwise distance ratios
    ratios = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d_da3 = np.linalg.norm(da3_centers[names[i]] - da3_centers[names[j]])
            d_colmap = np.linalg.norm(colmap_centers[names[i]] - colmap_centers[names[j]])
            if d_da3 > 1e-6 and d_colmap > 1e-6:
                ratios.append(d_colmap / d_da3)

    if not ratios:
        raise ValueError("No valid distance ratios computed")

    scale = float(np.median(ratios))
    print(f"  Computed scale factor (colmap/da3): {scale:.6f}")
    print(f"  Common images: {len(names)}, pairwise ratios: {len(ratios)}")
    return scale


def scale_colmap_output(src_dir, dst_dir, scale_factor):
    """
    Scale a COLMAP reconstruction by a given factor.

    Uses colmap_loader to read binary files (authoritative source),
    then writes scaled txt output and a CloudCompare-compatible PLY.

    Only changes: camera tvec, point XYZ. Everything else preserved.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # ---- Read COLMAP data via colmap_loader ----
    cameras = read_intrinsics_binary(os.path.join(src_dir, "cameras.bin"))
    images = read_extrinsics_binary(os.path.join(src_dir, "images.bin"))
    # read_points3D_binary returns (xyzs, rgbs, errors) but not IDs or track data
    # We need IDs and tracks for txt output, so read manually
    point_ids, xyzs, rgbs, errors, tracks = _read_points3D_full(
        os.path.join(src_dir, "points3D.bin")
    )

    # ---- Write scaled cameras.txt ----
    with open(os.path.join(dst_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cid, cam in sorted(cameras.items()):
            sp = list(cam.params)
            if cam.model == "PINHOLE":
                # fx, fy, cx, cy
                sp = [x * scale_factor for x in sp]
            else:
                # Scale focal length only
                sp[0] *= scale_factor
            params_str = " ".join(f"{x}" for x in sp)
            f.write(f"{cid} {cam.model} {cam.width} {cam.height} {params_str}\n")

    # ---- Write scaled images.txt ----
    with open(os.path.join(dst_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for img_id in sorted(images.keys()):
            img = images[img_id]
            tvec_s = img.tvec * scale_factor
            f.write(
                f"{img_id} {img.qvec[0]} {img.qvec[1]} {img.qvec[2]} {img.qvec[3]} "
                f"{tvec_s[0]} {tvec_s[1]} {tvec_s[2]} {img.camera_id} {img.name}\n"
            )
            # Write 2D observations: X Y POINT3D_ID (COLMAP text format)
            obs = []
            for j in range(len(img.xys)):
                x, y = img.xys[j]
                p3d_id = img.point3D_ids[j]
                obs.append(f"{x} {y} {p3d_id}")
            f.write(" ".join(obs) + "\n")

    # ---- Write scaled points3D.txt ----
    num_points = len(point_ids)
    with open(os.path.join(dst_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Compute mean track length for header
        total_tracks = sum(len(tr) for tr in tracks)
        mean_track = total_tracks / num_points if num_points > 0 else 0
        f.write(f"# Number of points: {num_points}, mean track length: {mean_track:.4f}\n")
        for i in range(num_points):
            pid = point_ids[i]
            xyz_s = xyzs[i] * scale_factor
            rgb = rgbs[i]
            err = errors[i, 0] if errors.ndim > 1 else errors[i]
            track = " ".join(f"{t[0]} {t[1]}" for t in tracks[i])
            f.write(
                f"{pid} {xyz_s[0]} {xyz_s[1]} {xyz_s[2]} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} {err} {track}\n"
            )

    print(f"  Scaled COLMAP output written to {dst_dir}")
    print(f"  Points: {num_points}, Cameras: {len(cameras)}, Images: {len(images)}")

    # ---- Export simple point cloud for CloudCompare (ASCII PLY) ----
    pcd_path = os.path.join(dst_dir, "point_cloud.ply")
    with open(pcd_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            xyz_s = xyzs[i] * scale_factor
            f.write(f"{xyz_s[0]} {xyz_s[1]} {xyz_s[2]} {rgbs[i][0]} {rgbs[i][1]} {rgbs[i][2]}\n")
    print(f"  Point cloud exported to {pcd_path}")


def _read_points3D_full(path):
    """
    Read COLMAP points3D.bin and return all data including IDs and track info.
    Returns: (point_ids, xyzs, rgbs, errors, tracks)
    """
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        point_ids = []
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3), dtype=np.uint8)
        errors = np.empty(num_points)
        tracks = []
        for i in range(num_points):
            pid = struct.unpack("<Q", f.read(8))[0]
            point_ids.append(pid)
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id = struct.unpack("<I", f.read(4))[0]
                pt2d_idx = struct.unpack("<I", f.read(4))[0]
                track.append((img_id, pt2d_idx))
            xyzs[i] = xyz
            rgbs[i] = rgb
            errors[i] = err
            tracks.append(track)
    return point_ids, xyzs, rgbs, errors, tracks


def run_da3_streaming(image_dir, output_dir, config_path):
    """Run DA3-Streaming as a separate process to free memory afterwards."""
    print("\n" + "=" * 60)
    print("Step 1: DA3-Streaming")
    print("=" * 60)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        sys.executable,
        os.path.join(CURRENT_DIR, "da3_streaming.py"),
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--config", config_path,
    ]
    result = subprocess.run(cmd, cwd=CURRENT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"DA3-Streaming failed with return code {result.returncode}")

    # Convert to COLMAP format (lightweight, no heavy imports needed)
    print("\nConverting DA3 output to COLMAP format...")
    sparse_dir = output2colmap(output_dir, image_dir)
    print(f"DA3-Streaming done. COLMAP output at: {sparse_dir}")


def run_colmap_recon(base_path):
    """Run COLMAP sparse reconstruction as a separate process."""
    print("\n" + "=" * 60)
    print("Step 2: COLMAP Sparse Reconstruction")
    print("=" * 60)

    cmd = [
        sys.executable,
        os.path.join(CURRENT_DIR, "colmap_sparse_recon.py"),
        base_path,
    ]
    result = subprocess.run(cmd, cwd=CURRENT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP sparse reconstruction failed with return code {result.returncode}")

    sparse_output_dir = os.path.join(base_path, "sparse", "0")
    if not os.path.exists(sparse_output_dir):
        raise RuntimeError(f"COLMAP reconstruction failed: {sparse_output_dir} not found")

    print(f"COLMAP sparse reconstruction done. Output at: {sparse_output_dir}")
    return sparse_output_dir


def run_scale_alignment(base_path):
    """Compute scale factor and apply it to COLMAP output."""
    print("\n" + "=" * 60)
    print("Step 3: Scale Alignment")
    print("=" * 60)

    da3_sparse_dir = os.path.join(base_path, "da3_sparse", "0")
    colmap_sparse_dir = os.path.join(base_path, "sparse", "0")
    colmap_backup_dir = os.path.join(base_path, "colmap_sparse", "0")
    final_sparse_dir = os.path.join(base_path, "sparse", "0")

    if not os.path.exists(colmap_sparse_dir):
        print("No COLMAP output found. Skipping scale alignment.")
        return

    # Compute scale factor
    scale = compute_scale_factor(da3_sparse_dir, colmap_sparse_dir)

    # Backup original COLMAP output
    if os.path.exists(colmap_backup_dir):
        shutil.rmtree(colmap_backup_dir)
    os.makedirs(os.path.join(base_path, "colmap_sparse"), exist_ok=True)
    shutil.move(colmap_sparse_dir, colmap_backup_dir)
    print(f"  Original COLMAP output moved to: {colmap_backup_dir}")

    # Write scaled COLMAP output to sparse/0
    scale_colmap_output(colmap_backup_dir, final_sparse_dir, scale)

    print(f"\n✅ Pipeline complete!")
    print(f"  DA3 output:      {os.path.dirname(os.path.dirname(da3_sparse_dir))}")
    print(f"  DA3 COLMAP:      {da3_sparse_dir}")
    print(f"  COLMAP (orig):   {colmap_backup_dir}")
    print(f"  COLMAP (scaled): {final_sparse_dir}")
    print(f"  Scale factor:    {scale:.6f}")


def run_pipeline(image_dir, output_dir, config_path, run_colmap=True):
    """
    Run the full pipeline. Each heavy step runs in a separate process
    to avoid memory accumulation.
    """
    base_path = get_base_path(image_dir)

    # Step 1: DA3-Streaming (separate process)
    run_da3_streaming(image_dir, output_dir, config_path)

    if not run_colmap:
        print("\nSkipping COLMAP reconstruction (--no_colmap).")
        return

    # Step 2: COLMAP (separate process - fresh memory)
    run_colmap_recon(base_path)

    # Step 3: Scale alignment (lightweight, runs in current process)
    run_scale_alignment(base_path)


def main():
    parser = argparse.ArgumentParser(
        description="DA3-Streaming + COLMAP pipeline with scale alignment"
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory path")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="DA3-Streaming output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=os.path.join(CURRENT_DIR, "configs/base_config.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--no_colmap",
        action="store_true",
        help="Skip COLMAP sparse reconstruction (use DA3 output only)",
    )
    args = parser.parse_args()

    image_dir = args.image_dir

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = os.path.join(CURRENT_DIR, "exps")
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    run_pipeline(
        image_dir=image_dir,
        output_dir=save_dir,
        config_path=args.config,
        run_colmap=not args.no_colmap,
    )


if __name__ == "__main__":
    main()
