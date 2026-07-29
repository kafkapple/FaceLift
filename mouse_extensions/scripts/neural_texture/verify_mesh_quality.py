"""Verify MAMMAL mesh OBJ quality via cam_003 projection IoU.

Projects mesh vertices onto camera 3 and compares with GT alpha mask.
Frames with IoU < 0.7 are flagged as bad (articulation errors).

Origin: Created during S32 neural texture PoC as /tmp/bad_frames.py.
Moved to repo in S33 (2026-03-23).

Usage:
    python -m mouse_extensions.scripts.neural_texture.verify_mesh_quality

Requires: trimesh, scipy, PIL
"""
import json, numpy as np, trimesh
from pathlib import Path
from scipy.ndimage import binary_dilation
from PIL import Image
from mouse_extensions.paths import M5_DATA

M5_SC=np.array([59.672,51.517,107.099]); M5_DS=2.7/307.785
m5=M5_DATA
obj_dir=Path("/home/joon/data/synthetic/textured_obj")

bad, good = [], []
for obj in sorted(obj_dir.glob("step_2_frame_*.obj")):
    mf=int(obj.stem.split("_")[-1]); m5f=mf//5
    cp=m5/("%06d"%m5f)/"opencv_cameras.json"
    if not cp.exists(): continue
    cd=json.load(open(cp))
    f=cd["frames"][3]
    w2c=np.array(f["w2c"]); fx,fy,cx,cy=f["fx"],f["fy"],f["cx"],f["cy"]
    mesh=trimesh.load(str(obj),process=False)
    verts=((mesh.vertices-M5_SC)*M5_DS)
    vh=np.hstack([verts,np.ones((len(verts),1))])
    cam=(w2c@vh.T).T
    px=fx*cam[:,0]/cam[:,2]+cx; py=fy*cam[:,1]/cam[:,2]+cy
    valid=(cam[:,2]>0)&(px>=0)&(px<512)&(py>=0)&(py<512)
    mask=np.zeros((512,512),bool)
    if valid.sum()>0:
        vx=np.clip(px[valid].astype(int),0,511)
        vy=np.clip(py[valid].astype(int),0,511)
        mask[vy,vx]=True
        mask=binary_dilation(mask,iterations=5)
    gt=np.array(Image.open(m5/("%06d"%m5f)/"images"/"cam_003.png"))
    gt_mask=gt[:,:,3]>128
    inter=(mask&gt_mask).sum(); union=(mask|gt_mask).sum()
    iou=inter/max(union,1)
    if iou < 0.7:
        bad.append((m5f, mf, iou))
    else:
        good.append((m5f, mf, iou))

print("Bad frames (IoU < 0.7):")
for m5f, mf, iou in sorted(bad, key=lambda x: x[2]):
    print("  M5=%4d MAMMAL=%5d IoU=%.3f" % (m5f, mf, iou))
print("Total bad: %d, good: %d" % (len(bad), len(good)))
mammal_bad = sorted([x[1] for x in bad])
print("\nMAMMAL frames to re-fit:", mammal_bad)
