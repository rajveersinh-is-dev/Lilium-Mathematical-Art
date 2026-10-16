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
    out = []
    for x in c[:3]:
        if x <= 0.04045:
            out.append(x / 12.92)
        else:
            out.append(((x + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def srgb_to_linear_rgba(c, alpha=1.0):
    """Converts sRGB [0..1] color tuple to scene-linear RGBA (4 elements)."""
    rgb = srgb_to_linear_rgb(c[:3])
    return (rgb[0], rgb[1], rgb[2], float(alpha))


def create_petal_material():
    """
    White Oriental Lily Petal Material (*Lilium 'Casa Blanca'*):
    - Satiny snowy ivory with realistic epidermal papillae and subtle celadon nectary groove
    - Random Walk Subsurface Scattering for living petal tissue transmission
    - Micro-papillae sheen without harsh plastic reflections
    """
    mat = bpy.data.materials.new(name="LilyPetalMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.26
    bsdf.inputs['IOR'].default_value = 1.44
    bsdf.inputs['Subsurface Weight'].default_value = 0.65
    bsdf.inputs['Subsurface Radius'].default_value = (0.38, 0.24, 0.11)
    bsdf.inputs['Subsurface Scale'].default_value = 0.025
    bsdf.inputs['Subsurface Anisotropy'].default_value = 0.40
    bsdf.inputs['Sheen Weight'].default_value = 0.65
    bsdf.inputs['Sheen Roughness'].default_value = 0.28
    bsdf.inputs['Sheen Tint'].default_value = srgb_to_linear_rgba((1.0, 0.99, 0.97))

    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])

    abs_t = nodes.new('ShaderNodeMath')
    abs_t.operation = 'ABSOLUTE'
    links.new(sep_uv.outputs['Y'], abs_t.inputs[0])

    col_base = srgb_to_linear_rgba((0.985, 0.982, 0.970))
    base_col_node = nodes.new('ShaderNodeRGB')
    base_col_node.outputs['Color'].default_value = col_base

    # Delicate translucent celadon-lime nectary groove (narrow, natural channel)
    col_nectary = srgb_to_linear_rgba((0.80, 0.92, 0.70))
    t_ramp = nodes.new('ShaderNodeValToRGB')
    t_ramp.color_ramp.elements[0].position = 0.015
    t_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    t_ramp.color_ramp.elements[1].position = 0.065
    t_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(abs_t.outputs['Value'], t_ramp.inputs['Fac'])

    s_ramp = nodes.new('ShaderNodeValToRGB')
    s_ramp.color_ramp.elements[0].position = 0.08
    s_ramp.color_ramp.elements[0].color = (0.2, 0.2, 0.2, 1.0)
    s_ramp.color_ramp.elements.new(0.18)
    s_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    s_ramp.color_ramp.elements[2].position = 0.32
    s_ramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    links.new(sep_uv.outputs['X'], s_ramp.inputs['Fac'])

    nec_mult = nodes.new('ShaderNodeMath')
    nec_mult.operation = 'MULTIPLY'
    links.new(t_ramp.outputs['Color'], nec_mult.inputs[0])
    links.new(s_ramp.outputs['Color'], nec_mult.inputs[1])

    mix_nec = nodes.new('ShaderNodeMix')
    mix_nec.data_type = 'RGBA'
    mix_nec.clamp_factor = True
    links.new(nec_mult.outputs['Value'], mix_nec.inputs[0])
    links.new(base_col_node.outputs['Color'], mix_nec.inputs[6])
    nec_col_node = nodes.new('ShaderNodeRGB')
    nec_col_node.outputs['Color'].default_value = col_nectary
    links.new(nec_col_node.outputs['Color'], mix_nec.inputs[7])

    links.new(mix_nec.outputs[2], bsdf.inputs['Base Color'])

    # Epidermal Papillae & Longitudinal Venation Bump:
    # 1. Subtle longitudinal venation bands
    wave_tex = nodes.new('ShaderNodeTexWave')
    wave_tex.wave_type = 'BANDS'
    wave_tex.bands_direction = 'Y'
    wave_tex.inputs['Scale'].default_value = 140.0
    wave_tex.inputs['Distortion'].default_value = 0.22
    wave_tex.inputs['Detail'].default_value = 2.0
    links.new(uv_node.outputs['UV'], wave_tex.inputs['Vector'])

    # 2. Fleshy epidermal papillae (Voronoi bumps) clustered along nectary groove
    voro_tex = nodes.new('ShaderNodeTexVoronoi')
    voro_tex.feature = 'F1'
    voro_tex.distance = 'EUCLIDEAN'
    voro_tex.inputs['Scale'].default_value = 110.0
    links.new(uv_node.outputs['UV'], voro_tex.inputs['Vector'])

    # Papillae mask: s in [0.08, 0.32], |t| < 0.16
    pap_s_ramp = nodes.new('ShaderNodeValToRGB')
    pap_s_ramp.color_ramp.elements[0].position = 0.08
    pap_s_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    pap_s_ramp.color_ramp.elements.new(0.16)
    pap_s_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    pap_s_ramp.color_ramp.elements[2].position = 0.32
    pap_s_ramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    links.new(sep_uv.outputs['X'], pap_s_ramp.inputs['Fac'])

    pap_t_ramp = nodes.new('ShaderNodeValToRGB')
    pap_t_ramp.color_ramp.elements[0].position = 0.04
    pap_t_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    pap_t_ramp.color_ramp.elements[1].position = 0.16
    pap_t_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(abs_t.outputs['Value'], pap_t_ramp.inputs['Fac'])

    pap_mask_mult = nodes.new('ShaderNodeMath')
    pap_mask_mult.operation = 'MULTIPLY'
    links.new(pap_s_ramp.outputs['Color'], pap_mask_mult.inputs[0])
    links.new(pap_t_ramp.outputs['Color'], pap_mask_mult.inputs[1])

    pap_inv = nodes.new('ShaderNodeMath')
    pap_inv.operation = 'SUBTRACT'
    pap_inv.inputs[0].default_value = 1.0
    links.new(voro_tex.outputs['Distance'], pap_inv.inputs[1])

    pap_bump_val = nodes.new('ShaderNodeMath')
    pap_bump_val.operation = 'MULTIPLY'
    links.new(pap_inv.outputs['Value'], pap_bump_val.inputs[0])
    links.new(pap_mask_mult.outputs['Value'], pap_bump_val.inputs[1])

    total_bump = nodes.new('ShaderNodeMath')
    total_bump.operation = 'ADD'
    links.new(wave_tex.outputs['Color'], total_bump.inputs[0])
    links.new(pap_bump_val.outputs['Value'], total_bump.inputs[1])

    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.032
    bump.inputs['Distance'].default_value = 0.001
    links.new(total_bump.outputs['Value'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_bud_material():
    """Fresh developing lily bud: jade green base transitioning to creamy ivory tip."""
    mat = bpy.data.materials.new(name="LilyBudMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.32
    bsdf.inputs['Subsurface Weight'].default_value = 0.60
    bsdf.inputs['Subsurface Radius'].default_value = (0.18, 0.30, 0.10)
    bsdf.inputs['Subsurface Scale'].default_value = 0.03

    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])

    col_bud_base = srgb_to_linear_rgba((0.28, 0.46, 0.20))
    col_bud_tip = srgb_to_linear_rgba((0.88, 0.90, 0.78))
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = col_bud_base
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = col_bud_tip
    links.new(sep_uv.outputs['X'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_filament_material():
    """Translucent slender succulent celery-green filament."""
    mat = bpy.data.materials.new(name="LilyFilamentMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.84, 0.94, 0.76))
    bsdf.inputs['Roughness'].default_value = 0.24
    bsdf.inputs['Subsurface Weight'].default_value = 0.65
    bsdf.inputs['Subsurface Radius'].default_value = (0.16, 0.28, 0.10)
    bsdf.inputs['Coat Weight'].default_value = 0.20
    bsdf.inputs['Coat Roughness'].default_value = 0.10
    return mat


def create_anther_material():
    """
    Velvety chocolate-cinnamon / amber-russet pollen:
    Eliminates artificial neon orange plastic look in favor of true botanical pollen texture.
    """
    mat = bpy.data.materials.new(name="LilyAntherMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.42, 0.22, 0.07))
    bsdf.inputs['Roughness'].default_value = 0.82
    bsdf.inputs['Sheen Weight'].default_value = 0.94
    bsdf.inputs['Sheen Roughness'].default_value = 0.65
    bsdf.inputs['Sheen Tint'].default_value = srgb_to_linear_rgba((0.55, 0.32, 0.12))

    tex = nodes.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 240.0
    tex.inputs['Detail'].default_value = 4.0
    tex.inputs['Roughness'].default_value = 0.75
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.05
    bump.inputs['Distance'].default_value = 0.001
    links.new(tex.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_pistil_material():
    """Celery-pistachio style tube with wet stigmatic nectar clearcoat."""
    mat = bpy.data.materials.new(name="LilyPistilMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.58, 0.76, 0.44))
    bsdf.inputs['Roughness'].default_value = 0.24
    bsdf.inputs['Subsurface Weight'].default_value = 0.60
    bsdf.inputs['Coat Weight'].default_value = 0.35
    bsdf.inputs['Coat Roughness'].default_value = 0.05
    return mat


def create_stem_material():
    """Botanical succulent stem and calyx receptacle."""
    mat = bpy.data.materials.new(name="LilyStemMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = srgb_to_linear_rgba((0.20, 0.38, 0.12))
    bsdf.inputs['Roughness'].default_value = 0.34
    bsdf.inputs['Subsurface Weight'].default_value = 0.45
    bsdf.inputs['Subsurface Radius'].default_value = (0.08, 0.18, 0.04)
    return mat


def create_leaf_material():
    """
    De-glossed, organic botanical foliage:
    Soft waxy cuticle without mirror clearcoat, chartreuse midrib groove, and chlorophyll SSS.
    """
    mat = bpy.data.materials.new(name="LilyLeafMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.42
    bsdf.inputs['Subsurface Weight'].default_value = 0.52
    bsdf.inputs['Subsurface Radius'].default_value = (0.08, 0.25, 0.04)
    bsdf.inputs['Subsurface Scale'].default_value = 0.035
    bsdf.inputs['Coat Weight'].default_value = 0.0  # ZERO MIRROR COAT!

    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = "UVMap"
    sep_uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(uv_node.outputs['UV'], sep_uv.inputs['Vector'])

    abs_t = nodes.new('ShaderNodeMath')
    abs_t.operation = 'ABSOLUTE'
    links.new(sep_uv.outputs['Y'], abs_t.inputs[0])

    col_leaf_midrib = srgb_to_linear_rgba((0.26, 0.42, 0.14))
    col_leaf_blade = srgb_to_linear_rgba((0.14, 0.28, 0.09))
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = col_leaf_midrib
    ramp.color_ramp.elements[1].position = 0.16
    ramp.color_ramp.elements[1].color = col_leaf_blade
    links.new(abs_t.outputs['Value'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])

    wave = nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Y'
    wave.inputs['Scale'].default_value = 140.0
    wave.inputs['Distortion'].default_value = 0.2
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.02
    bump.inputs['Distance'].default_value = 0.001
    links.new(wave.outputs['Color'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_backdrop(scene):
    """Muted botanical studio backdrop with natural chiaroscuro illumination falloff and depth."""
    bd_mesh = bpy.data.meshes.new("BackdropMesh")
    bw, bh, bz = 10.0, 7.0, -2.6
    bd_mesh.from_pydata([(-bw, -bh, bz), (bw, -bh, bz), (bw, bh, bz), (-bw, bh, bz)], [], [[0, 1, 2], [0, 2, 3]])
    bd_mesh.update()
    bd_obj = bpy.data.objects.new("Backdrop", bd_mesh)
    scene.collection.objects.link(bd_obj)

    bd_mat = bpy.data.materials.new("BackdropMat")
    bd_mat.use_nodes = True
    bd_nodes = bd_mat.node_tree.nodes
    bd_links = bd_mat.node_tree.links
    bd_nodes.clear()
    bd_out = bd_nodes.new('ShaderNodeOutputMaterial')
    bd_bsdf = bd_nodes.new('ShaderNodeBsdfPrincipled')
    bd_bsdf.inputs['Roughness'].default_value = 0.95

    bd_tc = bd_nodes.new('ShaderNodeTexCoord')
    bd_map = bd_nodes.new('ShaderNodeMapping')
    bd_links.new(bd_tc.outputs['Object'], bd_map.inputs['Vector'])
    bd_len = bd_nodes.new('ShaderNodeVectorMath')
    bd_len.operation = 'LENGTH'
    bd_links.new(bd_map.outputs['Vector'], bd_len.inputs[0])

    col_c = srgb_to_linear_rgba((0.20, 0.28, 0.22))
    col_e = srgb_to_linear_rgba((0.035, 0.052, 0.038))
    bd_ramp = bd_nodes.new('ShaderNodeValToRGB')
    bd_ramp.color_ramp.elements[0].position = 0.40
    bd_ramp.color_ramp.elements[0].color = col_c
    bd_ramp.color_ramp.elements[1].position = 2.40
    bd_ramp.color_ramp.elements[1].color = col_e
    bd_links.new(bd_len.outputs['Value'], bd_ramp.inputs['Fac'])
    bd_links.new(bd_ramp.outputs['Color'], bd_bsdf.inputs['Base Color'])
    bd_links.new(bd_bsdf.outputs['BSDF'], bd_out.inputs['Surface'])
    bd_mesh.materials.append(bd_mat)


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
        bg_node.inputs['Color'].default_value = srgb_to_linear_rgba((0.035, 0.048, 0.036))
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

    materials_list = [mat_petal, mat_bud, mat_fil, mat_anther, mat_pistil, mat_stem, mat_leaf]
    for m in materials_list:
        mesh.materials.append(m)

    mat_map = {1: 0, 2: 0, 8: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    for poly in mesh.polygons:
        v_idx = poly.vertices[0]
        m_id = int(vert_mat_ids[v_idx])
        poly.material_index = mat_map.get(m_id, 0)

    obj = bpy.data.objects.new("BouquetObject", mesh)
    scene.collection.objects.link(obj)

    # Studio Backdrop
    create_backdrop(scene)

    # Camera: 105mm Macro Prime Lens
    print("[2/4] Setting Photographic Macro Camera (105mm, F/14 DOF, AgX)...")
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens_unit = 'FOV'
    cam_data.angle = np.radians(36.0)
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 4.25
    cam_data.dof.aperture_fstop = 14.0

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = (0.0, 0.08, 4.65)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    def track_to(light, tgt):
        dir_v = Vector(tgt) - Vector(light.location)
        light.rotation_euler = dir_v.to_track_quat('-Z', 'Y').to_euler()

    tgt_center = (0.0, 0.08, 0.10)

    # Lighting: 4-Point Botanical Studio Softbox Rig
    print("[3/4] Configuring 4-Point Botanical Softbox Rig...")
    # 1. Key Octabox: Large 2.4m softbox at 45 deg left
    key = bpy.data.lights.new("KeySoftbox", type='AREA')
    key.energy = 450.0
    key.size = 2.4
    key.color = srgb_to_linear_rgb((1.0, 0.985, 0.96))
    key_obj = bpy.data.objects.new("KeySoftbox", key)
    key_obj.location = (-2.8, 2.0, 3.2)
    track_to(key_obj, tgt_center)
    scene.collection.objects.link(key_obj)

    # 2. Fill Softbox: 2.8m diffused softbox on right
    fill = bpy.data.lights.new("FillSoftbox", type='AREA')
    fill.energy = 115.0
    fill.size = 3.0
    fill.color = srgb_to_linear_rgb((0.92, 0.95, 1.0))
    fill_obj = bpy.data.objects.new("FillSoftbox", fill)
    fill_obj.location = (2.6, -0.1, 2.8)
    track_to(fill_obj, tgt_center)
    scene.collection.objects.link(fill_obj)

    # 3. Translucency Rim Backlight: at z = -1.2 (in front of backdrop at z = -2.6!)
    rim = bpy.data.lights.new("RimLight", type='AREA')
    rim.energy = 360.0
    rim.size = 2.0
    rim.color = srgb_to_linear_rgb((0.98, 1.0, 0.96))
    rim_obj = bpy.data.objects.new("RimLight", rim)
    rim_obj.location = (0.1, 2.4, -1.2)
    track_to(rim_obj, tgt_center)
    scene.collection.objects.link(rim_obj)

    # 4. Backdrop Halo Softbox (gently lighting backdrop from behind)
    halo = bpy.data.lights.new("BackdropHalo", type='SPOT')
    halo.energy = 850.0
    halo.spot_size = np.radians(85.0)
    halo.spot_blend = 0.55
    halo.color = srgb_to_linear_rgb((0.55, 0.70, 0.52))
    halo_obj = bpy.data.objects.new("BackdropHalo", halo)
    halo_obj.location = (0.0, 0.2, -0.8)
    track_to(halo_obj, (0.0, 0.08, -2.6))
    scene.collection.objects.link(halo_obj)

    # Render Output
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_p)

    print(f"[4/4] Path Tracing Bouquet via Cycles OptiX ({samples} samples)...")
    bpy.ops.render.render(write_still=True)
    t_tot = time.time() - t0
    print(f"[OK] Botanical Bouquet Successfully Rendered in {t_tot:.2f}s!")
    print(f"[OK] Saved: {out_p} ({width} x {height} px)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render Project Lilium Bouquet via Blender Cycles")
    parser.add_argument("--output", default="outputs/lily_bouquet.png")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=128)

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
