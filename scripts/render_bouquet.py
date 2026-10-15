"""
scripts/render_bouquet.py - Standalone script to synthesize the botanical lily bouquet render.

Supports dual rendering modes:
1. Production Photorealistic Path: Blender Cycles OptiX with Random Walk SSS and Chiaroscuro studio softbox lighting.
2. High-Performance Software Fallback: Native Numba-accelerated analytical Z-buffer rasterizer.
"""

import time
import json
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.foliage import BouquetScene
from src.renderer import LilyRenderer


def find_blender_binary():
    """Locates Blender executable if available."""
    # Check PATH first
    b_path = shutil.which("blender")
    if b_path:
        return b_path
        
    # Check standard Windows installation directories
    candidate_paths = [
        Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
    ]
    for p in candidate_paths:
        if p.exists():
            return str(p)
            
    return None


def main():
    print("=" * 70)
    print("PROJECT LILIUM — SYNTHESIZING BOTANICAL WHITE-LILY BOUQUET")
    print("=" * 70)
    
    config_path = Path("configs/final_scene.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "resolution": {"width": 2400, "height": 1440},
            "random_seed": 2026,
            "camera": {"fov_degrees": 36.0}
        }
        
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    seed = config["random_seed"]
    output_path = Path("outputs/lily_bouquet.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    blender_exe = find_blender_binary()
    
    if blender_exe:
        print(f"[Engine] Found Blender Cycles: {blender_exe}")
        print(f"[Engine] Dispatching Monte Carlo Path Tracing via OptiX GPU ({width}x{height}, 128 samples)...")
        script_path = Path(__file__).resolve().parent.parent / "src" / "blender_cycles_bouquet.py"
        cmd = [
            blender_exe,
            "--background",
            "--python",
            str(script_path),
            "--",
            "--output",
            str(output_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            str(seed),
            "--samples",
            "128"
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and output_path.exists():
            print(f"[OK] Master Cycles OptiX Botanical Render Complete: {output_path.resolve()}")
            return
        else:
            print("[Warning] Cycles render returned non-zero. Falling back to native analytical rasterizer...")
            
    # Native Numba software rasterizer fallback
    print(f"[Engine] Using Native Analytical Z-Buffer Rasterizer (Width: {width}, Height: {height})...")
    t0 = time.time()
    scene = BouquetScene(seed=seed)
    scene_geom = scene.generate_bouquet_geometry()
    
    cam_pos = config.get("camera", {}).get("position", [0.0, 0.08, 4.45])
    cam_target = config.get("camera", {}).get("target", [0.0, -0.06, 0.0])
    fov = config.get("camera", {}).get("fov_degrees", 36.0)
    renderer = LilyRenderer(width=width, height=height, fov_deg=fov, camera_pos=cam_pos, camera_target=cam_target)
    
    rgb_array = renderer.render(scene_geom)
    t_render = time.time() - t0
    print(f"      Native Rendering Completed in {t_render:.2f} seconds.")
    
    img = Image.fromarray(rgb_array)
    img.save(output_path, quality=95)
    print(f"[OK] Saved Botanical Render: {output_path.resolve()} ({width}x{height})")


if __name__ == "__main__":
    main()
