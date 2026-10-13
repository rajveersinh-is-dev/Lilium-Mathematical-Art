"""
scripts/render_bouquet.py - Standalone script to synthesize the botanical lily bouquet render.
"""

import time
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.foliage import BouquetScene
from src.renderer import LilyRenderer


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
            "camera": {"fov_degrees": 38.0}
        }
        
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    seed = config["random_seed"]
    fov = config["camera"]["fov_degrees"]
    
    print(f"[1/4] Initializing Bouquet Scene (Seed: {seed}, Flowers: 10, Leaves: 35+)...")
    t0 = time.time()
    scene = BouquetScene(seed=seed)
    
    print("[2/4] Generating 3D Parametric Geometry (Tepals, Anthers, Stems, Foliage)...")
    scene_geom = scene.generate_bouquet_geometry()
    n_verts = len(scene_geom["vertices"])
    n_faces = len(scene_geom["faces"])
    print(f"      Total 3D Mesh Vertices: {n_verts:,}")
    print(f"      Total Triangular Facets: {n_faces:,}")
    
    print(f"[3/4] Initializing Parallel Z-Buffer PBR Renderer ({width}x{height}, FOV: {fov} deg)...")
    renderer = LilyRenderer(width=width, height=height, fov_deg=fov)
    
    print("[4/4] Rasterizing and Shading Pixels...")
    rgb_array = renderer.render(scene_geom)
    t_render = time.time() - t0
    print(f"      Rendering Completed in {t_render:.2f} seconds.")
    
    output_path = Path("outputs/lily_bouquet.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb_array)
    img.save(output_path, quality=95)
    print(f"[OK] Saved Botanical Render: {output_path.resolve()} ({width}x{height})")


if __name__ == "__main__":
    main()
