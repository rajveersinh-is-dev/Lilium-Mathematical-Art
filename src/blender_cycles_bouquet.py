"""
src/blender_cycles_bouquet.py - Production Blender Cycles OptiX renderer for Project Lilium.

Synthesizes the botanical white oriental lily bouquet via Monte Carlo path tracing with
Random Walk Subsurface Scattering, botanical softbox studio lighting, and
105mm macro photographic depth of field.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import bpy
from mathutils import Vector

from src.foliage import BouquetScene


def srgb_to_linear_rgb(c):
    """Converts sRGB [0..1] color tuple to scene-linear RGB (3 elements)."""
    return tuple(pow(max(0.0, float(x)), 2.2) for x in c[:3])


def srgb_to_linear_rgba(c, alpha=1.0):
    """Converts sRGB [0..1] color tuple to scene-linear RGBA (4 elements)."""
    a = float(c[3]) if len(c) > 3 else float(alpha)
    return tuple(pow(max(0.0, float(x)), 2.2) for x in c[:3]) + (a,)


def create_petal_material():
    """
    White Oriental Lily Petal Material (*Lilium 'Casa Blanca'*):
    - Base ivory white with vivid celery-lime nectary furrow along midrib
    - Delicate diffused rose/blush flank
    - Raised tactile throat papillae nodules & deep crimson-maroon freckles
    - Random Walk Subsurface Scattering (warm living petal tissue transmission)
    - Satiny epidermal sheen (velvet micro-papillae)
    - Delicate longitudinal venation striations
    """
    mat = bpy.data.materials.new(name="LilyPetalMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.22
    bsdf.inputs['IOR'].default_value = 1.44
    bsdf.inputs['Subsurface Weight'].default_value = 0.56
    # Fleshy organic light transport through living petal tissue
    bsdf.inputs['Subsurface Radius'].default_value = (0.28, 0.14, 0.06)
    bsdf.inputs['Subsurface Scale'].default_value = 0.035
    bsdf.inputs['Subsurface Anisotropy'].default_value = 0.35
    bsdf.inputs['Sheen Weight'].default_value = 0.70
    bsdf.inputs['Sheen Roughness'].default_value = 0.26
    bsdf.inputs['Sheen Tint'].default_value = srgb_to_linear_rgba((1.0, 0.98, 0.95))
    
    # UV Map node: X = s in [0, 1] (longitudinal), Y = t in [-1, 1] (transverse)
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])
    
    math_abs_t = nodes.new('ShaderNodeMath')
    math_abs_t.operation = 'ABSOLUTE'
    links.new(sep_uv.outputs['Y'], math_abs_t.inputs[0])
    
    # 1. Base Ivory White
    col_ivory = srgb_to_linear_rgba((0.985, 0.980, 0.965))
    base_color_node = nodes.new('ShaderNodeRGB')
    base_color_node.outputs['Color'].default_value = col_ivory
    
    # 2. Celery Lime Nectary Furrow: Vivid green channel running along the midrib (|t| < 0.20, 0.0 < s < 0.52)
    col_nectary = srgb_to_linear_rgba((0.32, 0.82, 0.14))
    
    t_nectary_ramp = nodes.new('ShaderNodeValToRGB')
    t_nectary_ramp.color_ramp.elements[0].position = 0.08
    t_nectary_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    t_nectary_ramp.color_ramp.elements[1].position = 0.22
    t_nectary_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(math_abs_t.outputs['Value'], t_nectary_ramp.inputs['Fac'])
    
    s_nectary_ramp = nodes.new('ShaderNodeValToRGB')
    s_nectary_ramp.color_ramp.elements[0].position = 0.30
    s_nectary_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    s_nectary_ramp.color_ramp.elements[1].position = 0.52
    s_nectary_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(sep_uv.outputs['X'], s_nectary_ramp.inputs['Fac'])
    
    nectary_mult = nodes.new('ShaderNodeMath')
    nectary_mult.operation = 'MULTIPLY'
    links.new(t_nectary_ramp.outputs['Color'], nectary_mult.inputs[0])
    links.new(s_nectary_ramp.outputs['Color'], nectary_mult.inputs[1])
    
    mix_nectary = nodes.new('ShaderNodeMix')
    mix_nectary.data_type = 'RGBA'
    mix_nectary.clamp_factor = True
    links.new(nectary_mult.outputs['Value'], mix_nectary.inputs[0])
    links.new(base_color_node.outputs['Color'], mix_nectary.inputs[6])
    nectary_color_node = nodes.new('ShaderNodeRGB')
    nectary_color_node.outputs['Color'].default_value = col_nectary
    links.new(nectary_color_node.outputs['Color'], mix_nectary.inputs[7])
    
    # 3. Soft Rose Midrib Blush: Delicate flush radiating along the middle section
    col_blush = srgb_to_linear_rgba((0.96, 0.68, 0.78))
    
    t_blush_ramp = nodes.new('ShaderNodeValToRGB')
    t_blush_ramp.color_ramp.elements[0].position = 0.0
    t_blush_ramp.color_ramp.elements[0].color = (0.60, 0.60, 0.60, 1.0)
    t_blush_ramp.color_ramp.elements[1].position = 0.38
    t_blush_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(math_abs_t.outputs['Value'], t_blush_ramp.inputs['Fac'])
    
    s_blush_ramp = nodes.new('ShaderNodeValToRGB')
    s_blush_ramp.color_ramp.elements[0].position = 0.15
    s_blush_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    s_blush_ramp.color_ramp.elements[1].position = 0.35
    s_blush_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(sep_uv.outputs['X'], s_blush_ramp.inputs['Fac'])
    
    s_blush_fade = nodes.new('ShaderNodeValToRGB')
    s_blush_fade.color_ramp.elements[0].position = 0.55
    s_blush_fade.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    s_blush_fade.color_ramp.elements[1].position = 0.72
    s_blush_fade.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(sep_uv.outputs['X'], s_blush_fade.inputs['Fac'])
    
    blush_s_mult = nodes.new('ShaderNodeMath')
    blush_s_mult.operation = 'MULTIPLY'
    links.new(s_blush_ramp.outputs['Color'], blush_s_mult.inputs[0])
    links.new(s_blush_fade.outputs['Color'], blush_s_mult.inputs[1])
    
    blush_fac = nodes.new('ShaderNodeMath')
    blush_fac.operation = 'MULTIPLY'
    links.new(t_blush_ramp.outputs['Color'], blush_fac.inputs[0])
    links.new(blush_s_mult.outputs['Value'], blush_fac.inputs[1])
    
    mix_blush = nodes.new('ShaderNodeMix')
    mix_blush.data_type = 'RGBA'
    mix_blush.clamp_factor = True
    links.new(blush_fac.outputs['Value'], mix_blush.inputs[0])
    links.new(mix_nectary.outputs[2], mix_blush.inputs[6])
    blush_color_node = nodes.new('ShaderNodeRGB')
    blush_color_node.outputs['Color'].default_value = col_blush
    links.new(blush_color_node.outputs['Color'], mix_blush.inputs[7])
    
    # 4. Botanical Throat Papillae & Crimson Freckles
    # Raised fleshy nodules flanking the nectary channel (|t| in [0.03, 0.28], s in [0.12, 0.52])
    voro_papillae = nodes.new('ShaderNodeTexVoronoi')
    voro_papillae.inputs['Scale'].default_value = 46.0
    voro_papillae.feature = 'F1'
    links.new(uv_node.outputs['UV'], voro_papillae.inputs['Vector'])
    
    papillae_ramp = nodes.new('ShaderNodeValToRGB')
    papillae_ramp.color_ramp.elements[0].position = 0.0
    papillae_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    papillae_ramp.color_ramp.elements[1].position = 0.36
    papillae_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(voro_papillae.outputs['Distance'], papillae_ramp.inputs['Fac'])
    
    s_pap_mask = nodes.new('ShaderNodeValToRGB')
    s_pap_mask.color_ramp.elements[0].position = 0.12
    s_pap_mask.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    s_pap_mask.color_ramp.elements[1].position = 0.20
    s_pap_mask.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(sep_uv.outputs['X'], s_pap_mask.inputs['Fac'])
    
    s_pap_fade = nodes.new('ShaderNodeValToRGB')
    s_pap_fade.color_ramp.elements[0].position = 0.45
    s_pap_fade.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    s_pap_fade.color_ramp.elements[1].position = 0.54
    s_pap_fade.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(sep_uv.outputs['X'], s_pap_fade.inputs['Fac'])
    
    t_pap_mask = nodes.new('ShaderNodeValToRGB')
    t_pap_mask.color_ramp.elements[0].position = 0.02
    t_pap_mask.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    t_pap_mask.color_ramp.elements[1].position = 0.06
    t_pap_mask.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(math_abs_t.outputs['Value'], t_pap_mask.inputs['Fac'])
    
    t_pap_fade = nodes.new('ShaderNodeValToRGB')
    t_pap_fade.color_ramp.elements[0].position = 0.22
    t_pap_fade.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    t_pap_fade.color_ramp.elements[1].position = 0.28
    t_pap_fade.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(math_abs_t.outputs['Value'], t_pap_fade.inputs['Fac'])
    
    m1 = nodes.new('ShaderNodeMath')
    m1.operation = 'MULTIPLY'
    links.new(s_pap_mask.outputs['Color'], m1.inputs[0])
    links.new(s_pap_fade.outputs['Color'], m1.inputs[1])
    
    m2 = nodes.new('ShaderNodeMath')
    m2.operation = 'MULTIPLY'
    links.new(t_pap_mask.outputs['Color'], m2.inputs[0])
    links.new(t_pap_fade.outputs['Color'], m2.inputs[1])
    
    pap_mask_total = nodes.new('ShaderNodeMath')
    pap_mask_total.operation = 'MULTIPLY'
    links.new(m1.outputs['Value'], pap_mask_total.inputs[0])
    links.new(m2.outputs['Value'], pap_mask_total.inputs[1])
    
    papillae_final_fac = nodes.new('ShaderNodeMath')
    papillae_final_fac.operation = 'MULTIPLY'
    links.new(papillae_ramp.outputs['Color'], papillae_final_fac.inputs[0])
    links.new(pap_mask_total.outputs['Value'], papillae_final_fac.inputs[1])
    
    col_freckle = srgb_to_linear_rgba((0.48, 0.05, 0.08))
    mix_freckle = nodes.new('ShaderNodeMix')
    mix_freckle.data_type = 'RGBA'
    mix_freckle.clamp_factor = True
    links.new(papillae_final_fac.outputs['Value'], mix_freckle.inputs[0])
    links.new(mix_blush.outputs[2], mix_freckle.inputs[6])
    freckle_color_node = nodes.new('ShaderNodeRGB')
    freckle_color_node.outputs['Color'].default_value = col_freckle
    links.new(freckle_color_node.outputs['Color'], mix_freckle.inputs[7])
    
    links.new(mix_freckle.outputs[2], bsdf.inputs['Base Color'])
    
    # 5. Tactile 3D Papillae Nodules + Longitudinal Micro-Venation Bump
    wave = nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Y'
    wave.inputs['Scale'].default_value = 140.0
    wave.inputs['Distortion'].default_value = 0.4
    wave.inputs['Detail'].default_value = 2.0
    links.new(uv_node.outputs['UV'], wave.inputs['Vector'])
    
    wave_scaled = nodes.new('ShaderNodeMath')
    wave_scaled.operation = 'MULTIPLY'
    wave_scaled.inputs[1].default_value = 0.04
    links.new(wave.outputs['Color'], wave_scaled.inputs[0])
    
    pap_bump_scaled = nodes.new('ShaderNodeMath')
    pap_bump_scaled.operation = 'MULTIPLY'
    pap_bump_scaled.inputs[1].default_value = 0.30
    links.new(papillae_final_fac.outputs['Value'], pap_bump_scaled.inputs[0])
    
    bump_combine = nodes.new('ShaderNodeMath')
    bump_combine.operation = 'ADD'
    links.new(wave_scaled.outputs['Value'], bump_combine.inputs[0])
    links.new(pap_bump_scaled.outputs['Value'], bump_combine.inputs[1])
    
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.22
    bump.inputs['Distance'].default_value = 0.003
    links.new(bump_combine.outputs['Value'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_bud_material():
    """
    Fresh developing lily bud:
    - Succulent pale jade green calyx base transitioning to ivory along sepal sutures
    - Chlorophyll and fleshy tissue SSS
    - Waxy cuticular protective sheen
    - Longitudinal suture line bump
    """
    mat = bpy.data.materials.new(name="LilyBudMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.20
    bsdf.inputs['Subsurface Weight'].default_value = 0.62
    bsdf.inputs['Subsurface Radius'].default_value = (0.12, 0.22, 0.05)
    bsdf.inputs['Subsurface Scale'].default_value = 0.04
    bsdf.inputs['Coat Weight'].default_value = 0.40
    bsdf.inputs['Coat Roughness'].default_value = 0.08
    
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])
    
    # Gradient along bud length: jade green base -> creamy ivory tip
    col_base = srgb_to_linear_rgba((0.36, 0.62, 0.22))
    col_tip = srgb_to_linear_rgba((0.92, 0.95, 0.85))
    
    length_ramp = nodes.new('ShaderNodeValToRGB')
    length_ramp.color_ramp.elements[0].position = 0.15
    length_ramp.color_ramp.elements[0].color = col_base
    length_ramp.color_ramp.elements[1].position = 0.88
    length_ramp.color_ramp.elements[1].color = col_tip
    links.new(sep_uv.outputs['X'], length_ramp.inputs['Fac'])
    
    # Sepal suture margin highlight (|T| near 1.0 is ivory seam)
    math_abs_t = nodes.new('ShaderNodeMath')
    math_abs_t.operation = 'ABSOLUTE'
    links.new(sep_uv.outputs['Y'], math_abs_t.inputs[0])
    
    seam_ramp = nodes.new('ShaderNodeValToRGB')
    seam_ramp.color_ramp.elements[0].position = 0.70
    seam_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    seam_ramp.color_ramp.elements[1].position = 0.95
    seam_ramp.color_ramp.elements[1].color = (0.6, 0.6, 0.6, 1.0)
    links.new(math_abs_t.outputs['Value'], seam_ramp.inputs['Fac'])
    
    col_seam = srgb_to_linear_rgba((0.88, 0.92, 0.80))
    seam_mix = nodes.new('ShaderNodeMix')
    seam_mix.data_type = 'RGBA'
    seam_mix.clamp_factor = True
    links.new(seam_ramp.outputs['Color'], seam_mix.inputs[0])
    links.new(length_ramp.outputs['Color'], seam_mix.inputs[6])
    seam_color_node = nodes.new('ShaderNodeRGB')
    seam_color_node.outputs['Color'].default_value = col_seam
    links.new(seam_color_node.outputs['Color'], seam_mix.inputs[7])
    
    links.new(seam_mix.outputs[2], bsdf.inputs['Base Color'])
    
    # Longitudinal suture rib bump
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.06
    bump.inputs['Distance'].default_value = 0.002
    links.new(seam_ramp.outputs['Color'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_filament_material():
    """Translucent succulent celery green filament."""
    mat = bpy.data.materials.new(name="LilyFilamentMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.72, 0.88, 0.50))
    bsdf.inputs['Roughness'].default_value = 0.18
    bsdf.inputs['Subsurface Weight'].default_value = 0.70
    bsdf.inputs['Subsurface Radius'].default_value = (0.12, 0.20, 0.06)
    bsdf.inputs['Subsurface Scale'].default_value = 0.04
    return mat


def create_anther_material():
    """Deep velvety cinnamon-terracotta pollen with powdery diffuse roughness and micro-bump."""
    mat = bpy.data.materials.new(name="LilyAntherMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.68, 0.22, 0.02))
    bsdf.inputs['Diffuse Roughness'].default_value = 0.88
    bsdf.inputs['Roughness'].default_value = 0.74
    
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 220.0
    noise.inputs['Detail'].default_value = 5.0
    noise.inputs['Roughness'].default_value = 0.75
    
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.22
    bump.inputs['Distance'].default_value = 0.0015
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def create_pistil_material():
    """Pistachio style tube terminating in wet viscid stigmatic nectar clearcoat."""
    mat = bpy.data.materials.new(name="LilyPistilMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.52, 0.76, 0.34))
    bsdf.inputs['Roughness'].default_value = 0.20
    bsdf.inputs['Subsurface Weight'].default_value = 0.60
    bsdf.inputs['Subsurface Radius'].default_value = (0.10, 0.18, 0.05)
    bsdf.inputs['Subsurface Scale'].default_value = 0.05
    
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])
    
    coat_ramp = nodes.new('ShaderNodeValToRGB')
    coat_ramp.color_ramp.elements[0].position = 0.88
    coat_ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1.0)
    coat_ramp.color_ramp.elements[1].position = 0.98
    coat_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(sep_uv.outputs['X'], coat_ramp.inputs['Fac'])
    links.new(coat_ramp.outputs['Color'], bsdf.inputs['Coat Weight'])
    bsdf.inputs['Coat Roughness'].default_value = 0.03 # Glistening liquid stigma
    
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_stem_material():
    """Botanical succulent stem and calyx receptacle."""
    mat = bpy.data.materials.new(name="LilyStemMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.17, 0.36, 0.11))
    bsdf.inputs['Roughness'].default_value = 0.28
    bsdf.inputs['Subsurface Weight'].default_value = 0.48
    bsdf.inputs['Subsurface Radius'].default_value = (0.08, 0.15, 0.04)
    bsdf.inputs['Subsurface Scale'].default_value = 0.05
    return mat


def create_leaf_material():
    """Foliage leaf with chartreuse midrib, secondary parallel venation, and chlorophyll SSS."""
    mat = bpy.data.materials.new(name="LilyLeafMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.20
    bsdf.inputs['Subsurface Weight'].default_value = 0.62
    bsdf.inputs['Subsurface Radius'].default_value = (0.05, 0.18, 0.025)
    bsdf.inputs['Subsurface Scale'].default_value = 0.05
    bsdf.inputs['Coat Weight'].default_value = 0.35
    bsdf.inputs['Coat Roughness'].default_value = 0.12
    
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])
    
    math_abs_t = nodes.new('ShaderNodeMath')
    math_abs_t.operation = 'ABSOLUTE'
    links.new(sep_uv.outputs['Y'], math_abs_t.inputs[0])
    
    col_midrib = srgb_to_linear_rgba((0.28, 0.48, 0.14))
    col_blade = srgb_to_linear_rgba((0.07, 0.22, 0.05))
    
    leaf_ramp = nodes.new('ShaderNodeValToRGB')
    leaf_ramp.color_ramp.elements[0].position = 0.0
    leaf_ramp.color_ramp.elements[0].color = col_midrib
    leaf_ramp.color_ramp.elements[1].position = 0.12
    leaf_ramp.color_ramp.elements[1].color = col_blade
    links.new(math_abs_t.outputs['Value'], leaf_ramp.inputs['Fac'])
    links.new(leaf_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    leaf_wave = nodes.new('ShaderNodeTexWave')
    leaf_wave.wave_type = 'BANDS'
    leaf_wave.bands_direction = 'Y'
    leaf_wave.inputs['Scale'].default_value = 160.0
    leaf_wave.inputs['Distortion'].default_value = 0.3
    links.new(uv_node.outputs['UV'], leaf_wave.inputs['Vector'])
    
    leaf_bump = nodes.new('ShaderNodeBump')
    leaf_bump.inputs['Strength'].default_value = 0.035
    leaf_bump.inputs['Distance'].default_value = 0.002
    links.new(leaf_wave.outputs['Color'], leaf_bump.inputs['Height'])
    links.new(leaf_bump.outputs['Normal'], bsdf.inputs['Normal'])
    
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_raffia_material():
    """Natural golden hemp/raffia cord."""
    mat = bpy.data.materials.new(name="LilyRaffiaMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.58, 0.44, 0.26))
    bsdf.inputs['Roughness'].default_value = 0.75
    
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 240.0
    noise.inputs['Detail'].default_value = 4.0
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.28
    bump.inputs['Distance'].default_value = 0.002
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def create_backdrop(scene):
    """Muted botanical studio backdrop with natural chiaroscuro illumination falloff and depth."""
    plane_mesh = bpy.data.meshes.new("BackdropMesh")
    w, h, z = 7.5, 5.0, -1.8
    plane_verts = [(-w, -h + 0.1, z), (w, -h + 0.1, z), (w, h + 0.1, z), (-w, h + 0.1, z)]
    plane_faces = [[0, 1, 2], [0, 2, 3]]
    plane_mesh.from_pydata(plane_verts, [], plane_faces)
    plane_mesh.update()
    
    bd_obj = bpy.data.objects.new("Backdrop", plane_mesh)
    scene.collection.objects.link(bd_obj)
    
    mat = bpy.data.materials.new("BackdropMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.90
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
    
    math_len = nodes.new('ShaderNodeVectorMath')
    math_len.operation = 'LENGTH'
    links.new(mapping.outputs['Vector'], math_len.inputs[0])
    
    # Rich botanical deep forest moss / sage chiaroscuro vignette
    col_center = srgb_to_linear_rgba((0.14, 0.22, 0.16))
    col_edge = srgb_to_linear_rgba((0.028, 0.044, 0.032))
    
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.50
    ramp.color_ramp.elements[0].color = col_center
    ramp.color_ramp.elements[1].position = 2.60
    ramp.color_ramp.elements[1].color = col_edge
    links.new(math_len.outputs['Value'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    plane_mesh.materials.append(mat)


def render_bouquet_cycles(
    output_path: str = "outputs/lily_bouquet.png",
    width: int = 2400,
    height: int = 1440,
    seed: int = 2026,
    samples: int = 128
):
    """Renders the full bouquet using Blender Cycles OptiX."""
    t0 = time.time()
    print("=" * 70)
    print("PROJECT LILIUM — BLENDER CYCLES OPTIX BOTANICAL PATH TRACER")
    print(f"Target: {width} x {height} px | Samples: {samples} | Seed: {seed}")
    print("=" * 70)
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles
    cycles.device = 'GPU'
    cycles.samples = samples
    cycles.use_denoising = True
    
    # Configure GPU OptiX acceleration
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')
        
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    
    world = bpy.data.worlds.new("BotanicalWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = srgb_to_linear_rgba((0.045, 0.065, 0.050))
        bg_node.inputs['Strength'].default_value = 0.40
        
    print("[1/4] Synthesizing Parametric Bouquet Geometry from First Principles...")
    bouquet = BouquetScene(seed=seed)
    geom = bouquet.generate_bouquet_geometry()
    verts = geom["vertices"]
    faces = geom["faces"]
    uvs = geom["uvs"]
    vert_mat_ids = geom["mat_ids"]
    
    print(f"      3D Mesh Vertices: {len(verts):,}")
    print(f"      Triangular Facets: {len(faces):,}")
    
    mesh = bpy.data.meshes.new("BouquetMesh")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.shade_smooth()
    mesh.update()
    
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_vert_indices = [loop.vertex_index for loop in mesh.loops]
    for i, vi in enumerate(loop_vert_indices):
        uv_layer.data[i].uv = (float(uvs[vi, 0]), float(uvs[vi, 1]))
        
    mat_petal = create_petal_material()      # 0 -> mat_id 1, 2
    mat_bud = create_bud_material()          # 1 -> mat_id 8
    mat_fil = create_filament_material()     # 2 -> mat_id 3
    mat_anther = create_anther_material()    # 3 -> mat_id 4
    mat_pistil = create_pistil_material()    # 4 -> mat_id 5
    mat_stem = create_stem_material()        # 5 -> mat_id 6
    mat_leaf = create_leaf_material()        # 6 -> mat_id 7
    mat_raffia = create_raffia_material()    # 7 -> mat_id 9
    
    materials_list = [mat_petal, mat_bud, mat_fil, mat_anther, mat_pistil, mat_stem, mat_leaf, mat_raffia]
    for m in materials_list:
        mesh.materials.append(m)
        
    mat_map = {1: 0, 2: 0, 8: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7}
    for poly in mesh.polygons:
        v_idx = poly.vertices[0]
        m_id = int(vert_mat_ids[v_idx])
        poly.material_index = mat_map.get(m_id, 0)
        
    obj = bpy.data.objects.new("BouquetObject", mesh)
    scene.collection.objects.link(obj)
    
    # Studio Backdrop
    create_backdrop(scene)
    
    # Camera: 105mm Macro Botanical Setup
    print("[2/4] Setting Photographic Camera (105mm Macro, F/14 DOF, AgX)...")
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens_unit = 'FOV'
    cam_data.angle = np.radians(36.0)
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 4.22
    cam_data.dof.aperture_fstop = 14.0
    
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = (0.0, 0.08, 4.65)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    
    def track_to_point(light_obj, target_pos):
        loc = Vector(light_obj.location)
        tgt = Vector(target_pos)
        dir_vec = tgt - loc
        quat = dir_vec.to_track_quat('-Z', 'Y')
        light_obj.rotation_euler = quat.to_euler()
        
    target_center = (0.0, 0.08, 0.0)
    
    # Photographic Studio Lighting Setup
    print("[3/4] Configuring 3-Point Chiaroscuro Softbox Lighting...")
    # 1. Key Softbox: Large octabox placed high to the left (60 deg angle)
    key_data = bpy.data.lights.new(name="KeySoftbox", type='AREA')
    key_data.energy = 460.0
    key_data.size = 1.8
    key_data.color = srgb_to_linear_rgb((1.0, 0.985, 0.95))
    key_obj = bpy.data.objects.new("KeySoftbox", key_data)
    key_obj.location = (-3.0, 2.2, 3.0)
    track_to_point(key_obj, target_center)
    scene.collection.objects.link(key_obj)
    
    # 2. Fill Softbox: Diffused fill on the right
    fill_data = bpy.data.lights.new(name="FillSoftbox", type='AREA')
    fill_data.energy = 110.0
    fill_data.size = 2.4
    fill_data.color = srgb_to_linear_rgb((0.88, 0.93, 1.0))
    fill_obj = bpy.data.objects.new("FillSoftbox", fill_data)
    fill_obj.location = (2.8, -0.2, 2.6)
    track_to_point(fill_obj, target_center)
    scene.collection.objects.link(fill_obj)
    
    # 3. Translucency Rim Backlight: High behind bouquet
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 360.0
    rim_data.size = 1.5
    rim_data.color = srgb_to_linear_rgb((0.98, 1.00, 0.95))
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (0.2, 2.8, -2.2)
    track_to_point(rim_obj, target_center)
    scene.collection.objects.link(rim_obj)
    
    # 4. Backdrop Halo Softbox: Placed behind bouquet pointing at backdrop to create soft chiaroscuro halo
    halo_data = bpy.data.lights.new(name="BackdropHalo", type='AREA')
    halo_data.energy = 240.0
    halo_data.size = 2.8
    halo_data.color = srgb_to_linear_rgb((0.32, 0.52, 0.36))
    halo_obj = bpy.data.objects.new("BackdropHalo", halo_data)
    halo_obj.location = (0.0, 0.2, -0.6)
    track_to_point(halo_obj, (0.0, 0.2, -1.8))
    scene.collection.objects.link(halo_obj)
    
    # Render Output
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_p)
    
    print(f"[4/4] Monte Carlo Path Tracing via Cycles OptiX ({samples} samples)...")
    bpy.ops.render.render(write_still=True)
    t_tot = time.time() - t0
    print(f"[OK] Botanical Bouquet Successfully Rendered in {t_tot:.2f}s!")
    print(f"[OK] Saved: {out_p} ({width} x {height} px)")


if __name__ == "__main__":
    # Support command line invocation
    import argparse
    parser = argparse.ArgumentParser(description="Render Project Lilium Bouquet via Blender Cycles")
    parser.add_argument("--output", default="outputs/lily_bouquet.png")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=128)
    
    # Strip Blender CLI arguments if running inside blender
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    
    render_bouquet_cycles(
        output_path=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        samples=args.samples
    )
