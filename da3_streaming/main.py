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
    Image,
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


def scale_colmap_output(src_dir, dst_dir, s):
    """
    Scale a COLMAP reconstruction to match DA3's metric scale.

    s = da3_distance / colmap_distance (shrink factor, s < 1).
    Only changes: camera tvec *= s, point xyz *= s.
    Everything else preserved unchanged, output in same binary format as input.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Copy unchanged files: cameras.bin (intrinsics unchanged), project.ini
    shutil.copy2(os.path.join(src_dir, "cameras.bin"), dst_dir)
    project_ini = os.path.join(src_dir, "project.ini")
    if os.path.exists(project_ini):
        shutil.copy2(project_ini, dst_dir)

    # ---- Scale images: only tvec changes ----
    images = read_extrinsics_binary(os.path.join(src_dir, "images.bin"))
    scaled_images = {}
    for img_id, img in images.items():
        scaled_images[img_id] = Image(
            id=img.id, qvec=img.qvec, tvec=img.tvec * s,
            camera_id=img.camera_id, name=img.name,
            xys=img.xys, point3D_ids=img.point3D_ids,
        )
    _write_extrinsics_binary(os.path.join(dst_dir, "images.bin"), scaled_images)

    # ---- Scale points: only xyz changes ----
    point_ids, xyzs, rgbs, errors, tracks = _read_points3D_full(
        os.path.join(src_dir, "points3D.bin")
    )
    xyzs *= s
    _write_points3D_binary(
        os.path.join(dst_dir, "points3D.bin"), point_ids, xyzs, rgbs, errors, tracks
    )

    print(f"  Scaled COLMAP output written to {dst_dir}")
    print(f"  Points: {len(point_ids)}, Cameras: {len(images)}, Images: {len(images)}")
    print(f"  Scale (shrink factor): {s:.6f}")


def _write_extrinsics_binary(path, images):
    """Write COLMAP images.bin (mirrors read_extrinsics_binary format)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id in sorted(images.keys()):
            img = images[img_id]
            f.write(struct.pack("<idddddddi",
                                img.id,
                                img.qvec[0], img.qvec[1], img.qvec[2], img.qvec[3],
                                img.tvec[0], img.tvec[1], img.tvec[2],
                                img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(img.xys)))
            for j in range(len(img.xys)):
                f.write(struct.pack("<ddq",
                                    float(img.xys[j][0]), float(img.xys[j][1]),
                                    int(img.point3D_ids[j])))


def _write_points3D_binary(path, point_ids, xyzs, rgbs, errors, tracks):
    """Write COLMAP points3D.bin (mirrors _read_points3D_full format)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(point_ids)))
        for i in range(len(point_ids)):
            f.write(struct.pack("<Q", point_ids[i]))
            f.write(struct.pack("<3d", xyzs[i][0], xyzs[i][1], xyzs[i][2]))
            f.write(struct.pack("<3B", int(rgbs[i][0]), int(rgbs[i][1]), int(rgbs[i][2])))
            err_val = float(errors[i, 0] if errors.ndim > 1 else errors[i])
            f.write(struct.pack("<d", err_val))
            f.write(struct.pack("<Q", len(tracks[i])))
            for img_id, pt2d_idx in tracks[i]:
                f.write(struct.pack("<II", img_id, pt2d_idx))


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

    # Compute scale factor (colmap/da3 ratio) and shrink factor (da3/colmap)
    scale = compute_scale_factor(da3_sparse_dir, colmap_sparse_dir)
    s = 1.0 / scale

    # Backup original COLMAP output
    if os.path.exists(colmap_backup_dir):
        shutil.rmtree(colmap_backup_dir)
    os.makedirs(os.path.join(base_path, "colmap_sparse"), exist_ok=True)
    shutil.move(colmap_sparse_dir, colmap_backup_dir)
    print(f"  Original COLMAP output moved to: {colmap_backup_dir}")

    # Write scaled COLMAP output to sparse/0
    scale_colmap_output(colmap_backup_dir, final_sparse_dir, s)

    print(f"\n✅ Pipeline complete!")
    print(f"  DA3 output:      {os.path.dirname(os.path.dirname(da3_sparse_dir))}")
    print(f"  DA3 COLMAP:      {da3_sparse_dir}")
    print(f"  COLMAP (orig):   {colmap_backup_dir}")
    print(f"  COLMAP (scaled): {final_sparse_dir}")
    print(f"  Scale factor (colmap/da3): {scale:.6f}, shrink factor s = {s:.6f}")


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
