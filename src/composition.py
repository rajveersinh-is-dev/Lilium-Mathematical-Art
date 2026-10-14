"""
composition.py - Master artwork composer and layout assembler.

Features:
- Renders the standalone equation panel via Typst compiler.
- Assembles the square composition (2400 x 2400 px):
  - Upper botanical bouquet (2400 x 1440 px, 60% of canvas)
  - Elegant dividing caption band (2400 x 80 px)
  - Dense academic mathematical panel (2400 x 880 px, ~37% of canvas)
- Saves all standalone and composite deliverables in outputs/ and equations/.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import typst
from src.math_spec import generate_typst_panel_markup, generate_latex_source


def render_equation_panel(
    output_path: str = "outputs/lily_equations.png",
    typ_source_path: str = "equations/equation_panel.typ",
    target_width: int = 2400,
    target_height: int = 960
) -> Image.Image:
    """
    Compiles Typst mathematical specification to high-resolution PNG image
    and returns a PIL Image resized precisely to (target_width, target_height).
    """
    markup = generate_typst_panel_markup()
    Path(typ_source_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    Path(typ_source_path).write_text(markup, encoding="utf-8")
    
    temp_png = "outputs/temp_equation_render.png"
    # Typst compiles at 300 PPI for ultra-sharp typography
    typst.compile(typ_source_path, output=temp_png, format="png", ppi=300)
    
    eq_img = Image.open(temp_png).convert("RGB")
    
    # Resize to exact panel target dimensions
    resized_eq = eq_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    resized_eq.save(output_path)
    
    if os.path.exists(temp_png):
        os.remove(temp_png)
        
    return resized_eq


def render_caption_band(
    width: int = 2400,
    height: int = 80,
    title: str = "White Lilies by Rajveersinh Vishal Pardeshi",
    subtitle: str = "Parametric Perianth Manifolds · Trimerous Whorls · PBR Subsurface Translucency · 2400 × 1440 Procedural Synthesis"
) -> Image.Image:
    """
    Creates an academic caption band separating the botanical render and equation panel.
    Pure white background with sharp, elegant typography and subtle dividing borders.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Subtle top dividing hairline
    draw.line([(0, 0), (width, 0)], fill=(215, 220, 215), width=2)
    # Subtle bottom hairline
    draw.line([(0, height - 1), (width, height - 1)], fill=(230, 230, 230), width=1)
    
    # Fonts
    font_path_title = r"C:\Windows\Fonts\georgia.ttf"
    font_path_sub = r"C:\Windows\Fonts\segoeui.ttf"
    
    try:
        title_font = ImageFont.truetype(font_path_title, size=24)
        sub_font = ImageFont.truetype(font_path_sub, size=13)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
        
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (width - title_w) // 2
    title_y = 14
    draw.text((title_x, title_y), title, font=title_font, fill=(22, 25, 24))
    
    # Draw subtitle
    sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    sub_x = (width - sub_w) // 2
    sub_y = 48
    draw.text((sub_x, sub_y), subtitle, font=sub_font, fill=(90, 95, 92))
    
    return img


def compose_final_artwork(
    bouquet_image_path: str = "outputs/lily_bouquet.png",
    equations_image_path: str = "outputs/lily_equations.png",
    final_output_path: str = "outputs/lilium_mathematical_art.png",
    canvas_size: int = 2400,
    upper_height: int = 1440,
    caption_height: int = 0,
    equation_height: int = 960
) -> Image.Image:
    """
    Assembles the complete master artwork in square layout (canvas_size x canvas_size):
    Matching Hamid Naderi Yeganeh's artwork format:
    - [0 : upper_height] -> Botanical Bouquet image (2400 x 1440)
    - [upper_height : canvas_size] -> Seamless Mathematical Panel (2400 x 960)
    """
    bouquet_img = Image.open(bouquet_image_path).convert("RGB")
    if bouquet_img.size != (canvas_size, upper_height):
        bouquet_img = bouquet_img.resize((canvas_size, upper_height), Image.Resampling.LANCZOS)
        
    eq_img = Image.open(equations_image_path).convert("RGB")
    if eq_img.size != (canvas_size, equation_height):
        eq_img = eq_img.resize((canvas_size, equation_height), Image.Resampling.LANCZOS)
        
    master_canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    
    # Paste layers
    master_canvas.paste(bouquet_img, (0, 0))
    if caption_height > 0:
        caption_img = render_caption_band(width=canvas_size, height=caption_height)
        master_canvas.paste(caption_img, (0, upper_height))
        master_canvas.paste(eq_img, (0, upper_height + caption_height))
    else:
        master_canvas.paste(eq_img, (0, upper_height))
    
    Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
    master_canvas.save(final_output_path, quality=95)
    
    # Also save LaTeX source files
    latex_code = generate_latex_source()
    Path("equations/lilium_model.tex").write_text(latex_code, encoding="utf-8")
    Path("equations/equation_panel.tex").write_text(latex_code, encoding="utf-8")
    
    return master_canvas
