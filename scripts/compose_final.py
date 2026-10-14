"""
scripts/compose_final.py - Orchestration script to render bouquet, equation panel, and master art.
"""

import time
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.composition import compose_final_artwork


def main():
    print("=" * 70)
    print("PROJECT LILIUM — MASTER ARTWORK SYNTHESIS PIPELINE")
    print("=" * 70)
    t_start = time.time()
    
    # 1. Render Bouquet
    print("\n--- PHASE 1: BOTANICAL BOUQUET SYNTHESIS ---")
    ret_b = subprocess.run([sys.executable, "scripts/render_bouquet.py"], check=True)
    
    # 2. Render Equation Panel
    print("\n--- PHASE 2: MATHEMATICAL PANEL TYPESETTING ---")
    ret_e = subprocess.run([sys.executable, "scripts/render_equations.py"], check=True)
    
    # 3. Assemble Master Composition
    print("\n--- PHASE 3: COMPOSING MASTER ARTWORK ---")
    final_art = compose_final_artwork(
        bouquet_image_path="outputs/lily_bouquet.png",
        equations_image_path="outputs/lily_equations.png",
        final_output_path="outputs/lilium_mathematical_art.png",
        canvas_size=2400,
        upper_height=1440,
        caption_height=0,
        equation_height=960
    )
    
    total_time = time.time() - t_start
    print(f"\n[OK] Master Artwork Successfully Generated in {total_time:.2f}s!")
    print(f"[OK] Canvas Dimensions: {final_art.size[0]} x {final_art.size[1]} pixels (Square 1:1)")
    print(f"[OK] Output: {Path('outputs/lilium_mathematical_art.png').resolve()}")


if __name__ == "__main__":
    main()
