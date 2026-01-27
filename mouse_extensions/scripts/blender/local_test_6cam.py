"""
로컬 Blender GUI에서 실행:
1. Blender 열기
2. Scripting 탭 → New → 이 코드 붙여넣기 → Run Script

OBJ 경로를 수정 후 사용
"""

import bpy
import math
from mathutils import Vector

# ============================================================
# 설정 (여기서 수정)
# ============================================================
OBJ_PATH = "/Users/joon/Downloads/step_2_frame_000000.obj"  # 로컬 경로
NUM_CAMERAS = 6
CAMERA_DISTANCE = 2.7
ELEVATION = 20
FX = 549
IMAGE_SIZE = 512

# ============================================================
# 1. 씬 초기화
# ============================================================
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# ============================================================
# 2. OBJ 로드 + 정규화
# ============================================================
bpy.ops.wm.obj_import(filepath=OBJ_PATH)
obj = bpy.context.selected_objects[0]
obj.name = "Mouse"

# 정점 좌표 분석
import numpy as np
verts = np.array([[v.co.x, v.co.y, v.co.z] for v in obj.data.vertices])
center = verts.mean(axis=0)
size = verts.max(axis=0) - verts.min(axis=0)
max_dim = size.max()

print(f"Original: center={center}, size={size}, max_dim={max_dim:.1f}mm")

# 중심 이동 + mm→unit + 스케일
TARGET_SIZE = 1.5
scale = TARGET_SIZE / (max_dim * 0.001)

for v in obj.data.vertices:
    v.co.x = (v.co.x - center[0]) * 0.001 * scale
    v.co.y = (v.co.y - center[1]) * 0.001 * scale
    v.co.z = (v.co.z - center[2]) * 0.001 * scale

# MAMMAL 좌표계 보정 (-Z up → Z up)
# 현재: X=body length, Y=height?, Z=width?
# 이 부분을 수동으로 조정하면서 테스트!
# 옵션 A: Y, Z 반전 (180도 X축 회전)
for v in obj.data.vertices:
    v.co.y = -v.co.y
    v.co.z = -v.co.z

obj.data.update()

# 최종 크기 확인
verts_new = np.array([[v.co.x, v.co.y, v.co.z] for v in obj.data.vertices])
print(f"Final: size={verts_new.max(axis=0) - verts_new.min(axis=0)}")

# 간단한 재질
mat = bpy.data.materials.new("MouseMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.65, 0.55, 0.45, 1)
bsdf.inputs["Roughness"].default_value = 0.7
obj.data.materials.append(mat)

# ============================================================
# 3. 6개 카메라 생성
# ============================================================
cameras = []
for i in range(NUM_CAMERAS):
    azim_deg = i * (360.0 / NUM_CAMERAS)
    azim = math.radians(azim_deg)
    elev = math.radians(ELEVATION)
    
    x = CAMERA_DISTANCE * math.cos(elev) * math.sin(azim)
    y = -CAMERA_DISTANCE * math.cos(elev) * math.cos(azim)
    z = CAMERA_DISTANCE * math.sin(elev)
    
    cam_data = bpy.data.cameras.new(f"Cam_{i}")
    cam_obj = bpy.data.objects.new(f"Cam_{i}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    cam_obj.location = (x, y, z)
    
    # 원점을 바라보도록 설정
    direction = -cam_obj.location.normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # 초점 거리
    cam_data.lens = FX * 36 / IMAGE_SIZE  # ~38.6mm
    cam_data.sensor_width = 36
    
    cameras.append(cam_obj)
    print(f"Cam_{i}: azimuth={azim_deg:.0f}°, pos=({x:.2f}, {y:.2f}, {z:.2f})")

# ============================================================
# 4. 좌표축 표시 (참조용)
# ============================================================
# X축 (빨강) - 0.5 단위
bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=1, location=(0.5, 0, 0))
x_ax = bpy.context.active_object
x_ax.name = "X_Axis"
x_ax.rotation_euler = (0, math.radians(90), 0)
mat_r = bpy.data.materials.new("Red")
mat_r.diffuse_color = (1, 0, 0, 1)
x_ax.data.materials.append(mat_r)

# Y축 (초록)
bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=1, location=(0, 0.5, 0))
y_ax = bpy.context.active_object
y_ax.name = "Y_Axis"
y_ax.rotation_euler = (math.radians(90), 0, 0)
mat_g = bpy.data.materials.new("Green")
mat_g.diffuse_color = (0, 1, 0, 1)
y_ax.data.materials.append(mat_g)

# Z축 (파랑)
bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=1, location=(0, 0, 0.5))
z_ax = bpy.context.active_object
z_ax.name = "Z_Axis"
mat_b = bpy.data.materials.new("Blue")
mat_b.diffuse_color = (0, 0, 1, 1)
z_ax.data.materials.append(mat_b)

# ============================================================
# 5. 조명
# ============================================================
light = bpy.data.lights.new("Sun", "SUN")
light.energy = 3
light_obj = bpy.data.objects.new("Sun", light)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = (3, -3, 5)

# ============================================================
# 6. 첫 번째 카메라를 활성 카메라로 설정
# ============================================================
bpy.context.scene.camera = cameras[0]
bpy.context.scene.render.resolution_x = IMAGE_SIZE
bpy.context.scene.render.resolution_y = IMAGE_SIZE

print("\n" + "="*50)
print("Setup complete!")
print("- Numpad 0: 카메라 뷰")
print("- Ctrl+Numpad 0: 현재 뷰를 카메라로")
print("- Outliner에서 Cam_0~5 선택 후 Numpad 0으로 전환")
print("")
print("방향 수정 테스트:")
print("  obj.rotation_euler = (math.radians(X), math.radians(Y), math.radians(Z))")
print("  예: obj.rotation_euler = (math.radians(90), 0, 0)  # X축 90도")
print("="*50)
