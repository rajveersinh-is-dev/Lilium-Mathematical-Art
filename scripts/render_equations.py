"""
scripts/render_equations.py - Standalone script to compile the mathematical panel.
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.composition import render_equation_panel


def main():
    print("=" * 70)
    print("PROJECT LILIUM — TYPESETTING ANALYTICAL MATHEMATICAL PANEL")
    print("=" * 70)
    
    t0 = time.time()
    out_file = "outputs/lily_equations.png"
    print(f"[1/2] Compiling Typst mathematical specifications at 300 PPI...")
    panel_img = render_equation_panel(
        output_path=out_file,
        typ_source_path="equations/equation_panel.typ",
        target_width=2400,
        target_height=960
    )
    t_comp = time.time() - t0
    print(f"[2/2] Compiled and formatted in {t_comp:.2f} seconds.")
    print(f"[OK] Saved Equation Panel: {Path(out_file).resolve()} ({panel_img.size[0]}x{panel_img.size[1]})")


if __name__ == "__main__":
    main()
