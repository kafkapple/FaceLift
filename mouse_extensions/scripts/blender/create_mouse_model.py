#!/usr/bin/env python3
"""
create_mouse_model.py

마우스 유사 비대칭 3D 모델 생성
낮은 coverage (~5%), 비정형 형상

Usage:
    blender --background -P create_mouse_model.py -- --output model.blend
"""

import bpy
import math
import sys
import argparse
from mathutils import Vector


def clear_scene():
    """씬 초기화"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # 기본 컬렉션만 유지
    for collection in bpy.data.collections:
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)


def create_material(name, color):
    """단순 diffuse 머티리얼 생성"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Metallic"].default_value = 0.0
    
    # Output
    output = nodes.new("ShaderNodeOutputMaterial")
    
    # Link
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    
    return mat


def create_mouse_body():
    """마우스 몸통 (타원체)"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        segments=32,
        ring_count=16,
        location=(0, 0, 0)
    )
    body = bpy.context.active_object
    body.name = "Body"
    
    # 타원형으로 변형 (길쭉하게)
    body.scale = (1.8, 1.0, 0.8)
    bpy.ops.object.transform_apply(scale=True)
    
    return body


def create_mouse_head():
    """마우스 머리"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.5,
        segments=24,
        ring_count=12,
        location=(1.5, 0, 0.2)
    )
    head = bpy.context.active_object
    head.name = "Head"
    
    # 약간 납작하게
    head.scale = (1.2, 0.9, 0.85)
    bpy.ops.object.transform_apply(scale=True)
    
    return head


def create_mouse_ears():
    """마우스 귀 (2개)"""
    ears = []
    
    for side in [-1, 1]:  # 좌우
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.25,
            segments=16,
            ring_count=8,
            location=(1.7, side * 0.35, 0.5)
        )
        ear = bpy.context.active_object
        ear.name = f"Ear_{L if side < 0 else R}"
        ear.scale = (0.3, 0.8, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        ears.append(ear)
    
    return ears


def create_mouse_tail():
    """마우스 꼬리 (곡선 원뿔)"""
    # 기본 원뿔
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.15,
        radius2=0.02,
        depth=2.5,
        location=(-2.0, 0, 0)
    )
    tail = bpy.context.active_object
    tail.name = "Tail"
    
    # X축 방향으로 회전
    tail.rotation_euler = (0, math.radians(90), 0)
    bpy.ops.object.transform_apply(rotation=True)
    
    # 곡선 효과 (Lattice deform 대신 간단한 비틀기)
    # 실제로는 modifier 사용 권장
    
    return tail


def create_mouse_legs():
    """마우스 다리 (4개)"""
    legs = []
    
    # 앞다리 위치
    front_positions = [(0.8, -0.5, -0.5), (0.8, 0.5, -0.5)]
    # 뒷다리 위치
    back_positions = [(-0.8, -0.6, -0.5), (-0.8, 0.6, -0.5)]
    
    for i, pos in enumerate(front_positions + back_positions):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.1,
            depth=0.4,
            location=pos
        )
        leg = bpy.context.active_object
        leg.name = f"Leg_{i}"
        leg.scale = (0.8, 0.8, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        legs.append(leg)
    
    return legs


def join_objects(objects, name):
    """객체들을 하나로 합치기"""
    bpy.ops.object.select_all(action="DESELECT")
    
    for obj in objects:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    
    result = bpy.context.active_object
    result.name = name
    
    return result


def normalize_to_size(obj, target_size):
    """모델 크기 정규화"""
    # 원점 중심으로
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
    obj.location = (0, 0, 0)
    
    # 현재 크기
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    current_size = max(
        max(v.x for v in bbox) - min(v.x for v in bbox),
        max(v.y for v in bbox) - min(v.y for v in bbox),
        max(v.z for v in bbox) - min(v.z for v in bbox)
    )
    
    # 스케일링
    scale = target_size / current_size
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)
    
    print(f"Normalized: {current_size:.3f} -> {target_size:.3f}")
    print(f"Estimated coverage: ~{(target_size/2)**2 * 100:.1f}%")


def create_mouse_model(target_size=0.25):
    """전체 마우스 모델 생성"""
    
    print("\n" + "="*50)
    print("Creating Mouse-like Model")
    print("="*50 + "\n")
    
    clear_scene()
    
    # 부품 생성
    parts = []
    
    print("Creating body...")
    body = create_mouse_body()
    parts.append(body)
    
    print("Creating head...")
    head = create_mouse_head()
    parts.append(head)
    
    print("Creating ears...")
    ears = create_mouse_ears()
    parts.extend(ears)
    
    print("Creating tail...")
    tail = create_mouse_tail()
    parts.append(tail)
    
    print("Creating legs...")
    legs = create_mouse_legs()
    parts.extend(legs)
    
    # 합치기
    print("Joining parts...")
    mouse = join_objects(parts, "MouseModel")
    
    # 스무딩
    bpy.ops.object.shade_smooth()
    
    # 크기 정규화
    print("Normalizing size...")
    normalize_to_size(mouse, target_size)
    
    # 머티리얼
    print("Applying material...")
    mat = create_material("MouseMaterial", (0.6, 0.5, 0.4))  # 갈색
    mouse.data.materials.append(mat)
    
    # 삼각화 (렌더링 안정성)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    
    print(f"\nModel created: {mouse.name}")
    print(f"Vertices: {len(mouse.data.vertices)}")
    print(f"Faces: {len(mouse.data.polygons)}")
    
    return mouse


def setup_world():
    """월드 설정 (배경)"""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    
    bpy.context.scene.world = world
    world.use_nodes = True
    
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (1, 1, 1, 1)  # 흰 배경


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mouse_model.blend")
    parser.add_argument("--size", type=float, default=0.25)
    
    # Blender 인자 처리
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    args = parser.parse_args(argv)
    
    # 모델 생성
    mouse = create_mouse_model(args.size)
    setup_world()
    
    # 저장
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"\nSaved: {args.output}")
    
    print("\n" + "="*50)
    print("Model creation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
