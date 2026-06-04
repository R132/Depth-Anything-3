import os
import subprocess
import argparse


def run_colmap_sparse(
    input_path: str,
    camera_model: str = "PINHOLE",
    single_camera: bool = True
) -> None:
    # ===================== 路径配置 =====================
    image_dir = os.path.join(input_path, "images")
    database_path = os.path.join(input_path, "database.db")
    
    # 关键：sparse 根目录，mapper 会自动创建 0/ 文件夹在里面
    sparse_root_dir = os.path.join(input_path, "sparse")
    sparse_output_dir = os.path.join(sparse_root_dir, "0")

    # 自动创建目录
    os.makedirs(sparse_root_dir, exist_ok=True)

    # ===================== 检查图片 =====================
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    print("=" * 60)
    print(f"工作目录：{input_path}")
    print(f"图片目录：{image_dir}")
    print(f"输出目录：{sparse_output_dir}")
    print("=" * 60)

    # ===================== 1. 特征提取 =====================
    print("\n[1/5] 特征提取...")
    feat_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.use_gpu", "0",
    ]
    subprocess.run(feat_cmd, check=True)

    # ===================== 2. 特征匹配 =====================
    print("\n[2/5] 特征匹配...")
    match_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "0",
    ]
    subprocess.run(match_cmd, check=True)

    # ===================== 3. 稀疏重建（核心修复） =====================
    print("\n[3/5] 稀疏重建中...")

    # 正确用法：output_path = sparse 目录，mapper 自动生成 0/
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_root_dir,  # 这里必须是 sparse/，不能是 sparse/0
    ]
    subprocess.run(mapper_cmd, check=True)

    # ===================== 4. 检查是否生成成功 =====================
    if not os.path.exists(sparse_output_dir):
        raise RuntimeError(f"重建失败：未生成 {sparse_output_dir}")

    # ===================== 5. 转 TXT =====================
    print("\n[4/5] 导出 TXT ...")
    txt_cmd = [
        "colmap", "model_converter",
        "--input_path", sparse_output_dir,
        "--output_path", sparse_output_dir,
        "--output_type", "TXT",
    ]
    subprocess.run(txt_cmd, check=True)

    # ===================== 6. 导出 PLY 点云 =====================
    print("\n[5/5] 导出 PLY 点云...")
    ply_path = os.path.join(sparse_output_dir, "point_cloud.ply")
    ply_cmd = [
        "colmap", "model_converter",
        "--input_path", sparse_output_dir,
        "--output_path", ply_path,
        "--output_type", "PLY",
    ]
    subprocess.run(ply_cmd, check=True)

    print("\n✅ 全部完成！")
    print(f"📄 稀疏结果：{sparse_output_dir}")
    print(f"☁️  PLY点云：{ply_path}")


def main():
    parser = argparse.ArgumentParser(description="COLMAP 稀疏重建 - 最终修复版（输出到 sparse/0）")
    parser.add_argument("input_path", type=str, help="工作目录，里面放 images 文件夹")
    args = parser.parse_args()
    run_colmap_sparse(args.input_path)


if __name__ == "__main__":
    main()
