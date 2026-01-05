#!/usr/bin/env python3
"""
Human vs Mouse Camera Specification Comparison Script

인간 데이터 샘플과 생쥐 데이터 샘플의 카메라 파라미터를 정량적/정성적으로 비교하여
GS-LRM 학습 시 정규화가 올바르게 수행되는지 검증합니다.

Usage:
    python scripts/compare_human_mouse_camera_specs.py

Output:
    docs/reports/251218_camera_spec_comparison_report.md
    outputs/camera_comparison/

Author: AI-assisted (Claude)
Date: 2024-12-18
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Add FaceLift root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_opencv_cameras(json_path: str) -> dict:
    """Load camera data from opencv_cameras.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def analyze_cameras(frames: list, name: str) -> dict:
    """
    Analyze camera parameters from frames list.

    Returns:
        dict with camera statistics
    """
    results = {
        'name': name,
        'num_views': len(frames),
        'image_size': (frames[0]['w'], frames[0]['h']),
        'intrinsics': {
            'fx': frames[0]['fx'],
            'fy': frames[0]['fy'],
            'cx': frames[0]['cx'],
            'cy': frames[0]['cy'],
        },
        'cameras': [],
    }

    c2w_matrices = []
    for frame in frames:
        w2c = np.array(frame['w2c'])
        c2w = np.linalg.inv(w2c)
        c2w_matrices.append(c2w)

        # Extract camera info
        cam_pos = c2w[:3, 3]
        distance = np.linalg.norm(cam_pos)

        # Camera orientation (Y-down in OpenCV means -Y is camera up)
        cam_up = -c2w[:3, 1]  # Camera up vector in world coords
        cam_forward = c2w[:3, 2]  # Camera forward (Z) in world coords

        # Elevation and azimuth (assuming origin-centered)
        if distance > 1e-6:
            # Spherical coordinates
            # Azimuth: angle in XZ plane (or XY depending on up)
            azimuth = np.degrees(np.arctan2(cam_pos[0], cam_pos[2]))
            # Elevation: angle from horizontal plane
            elevation = np.degrees(np.arcsin(cam_pos[1] / distance))
        else:
            azimuth, elevation = 0, 0

        results['cameras'].append({
            'position': cam_pos.tolist(),
            'distance': float(distance),
            'elevation': float(elevation),
            'azimuth': float(azimuth),
            'up_vector': cam_up.tolist(),
            'forward_vector': cam_forward.tolist(),
        })

    # Aggregate statistics
    distances = [c['distance'] for c in results['cameras']]
    elevations = [c['elevation'] for c in results['cameras']]

    results['stats'] = {
        'distance_mean': float(np.mean(distances)),
        'distance_std': float(np.std(distances)),
        'distance_min': float(np.min(distances)),
        'distance_max': float(np.max(distances)),
        'elevation_mean': float(np.mean(elevations)),
        'elevation_std': float(np.std(elevations)),
        'elevation_min': float(np.min(elevations)),
        'elevation_max': float(np.max(elevations)),
    }

    # Check Y-up alignment
    up_vectors = np.array([c['up_vector'] for c in results['cameras']])
    avg_up = np.mean(up_vectors, axis=0)
    avg_up_normalized = avg_up / np.linalg.norm(avg_up)

    results['up_analysis'] = {
        'avg_up_vector': avg_up_normalized.tolist(),
        'y_component': float(avg_up_normalized[1]),
        'is_y_up': bool(avg_up_normalized[1] > 0.9),
        'is_y_down': bool(avg_up_normalized[1] < -0.9),
    }

    # Check orbit plane (cameras should lie roughly on a plane)
    positions = np.array([c['position'] for c in results['cameras']])
    centered = positions - np.mean(positions, axis=0)

    if len(positions) >= 3:
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        sorted_idx = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues.real[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Orbit normal is eigenvector with smallest eigenvalue
        orbit_normal = eigenvectors[:, 2].real
        planarity = 1.0 - eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 1e-6 else 0

        results['orbit_analysis'] = {
            'orbit_normal': orbit_normal.tolist(),
            'planarity': float(planarity),  # 1.0 = perfect plane
            'eigenvalues': eigenvalues.tolist(),
        }

    return results


def compare_results(human: dict, mouse: dict) -> dict:
    """Compare human and mouse camera analysis results."""
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'details': {},
        'issues': [],
        'recommendations': [],
    }

    # Image size comparison
    if human['image_size'] != mouse['image_size']:
        comparison['issues'].append(
            f"Image size mismatch: Human={human['image_size']}, Mouse={mouse['image_size']}"
        )
    else:
        comparison['summary']['image_size'] = "MATCH"

    # Intrinsics comparison
    h_intr = human['intrinsics']
    m_intr = mouse['intrinsics']

    fx_diff = abs(h_intr['fx'] - m_intr['fx'])
    if fx_diff < 1.0:
        comparison['summary']['focal_length'] = "MATCH"
    else:
        comparison['issues'].append(
            f"Focal length difference: Human fx={h_intr['fx']:.2f}, Mouse fx={m_intr['fx']:.2f}"
        )

    # Distance comparison
    h_dist = human['stats']['distance_mean']
    m_dist = mouse['stats']['distance_mean']
    dist_ratio = m_dist / h_dist if h_dist > 0 else 0

    comparison['details']['distance'] = {
        'human_mean': h_dist,
        'mouse_mean': m_dist,
        'ratio': dist_ratio,
        'target': 2.7,
    }

    if abs(m_dist - 2.7) < 0.1:
        comparison['summary']['camera_distance'] = "MATCH (≈2.7)"
    elif abs(m_dist - 2.7) < 0.3:
        comparison['summary']['camera_distance'] = "CLOSE"
    else:
        comparison['issues'].append(
            f"Camera distance not normalized: Mouse={m_dist:.3f}, Expected=2.7"
        )
        comparison['recommendations'].append(
            "Verify normalize_camera_distance() is being applied in MouseViewDataset"
        )

    # Up direction comparison
    h_up = human['up_analysis']
    m_up = mouse['up_analysis']

    comparison['details']['up_direction'] = {
        'human_avg_up': h_up['avg_up_vector'],
        'mouse_avg_up': m_up['avg_up_vector'],
        'human_y_component': h_up['y_component'],
        'mouse_y_component': m_up['y_component'],
    }

    if h_up['is_y_up'] and m_up['is_y_up']:
        comparison['summary']['up_direction'] = "MATCH (Y-up)"
    elif h_up['is_y_down'] and m_up['is_y_down']:
        comparison['summary']['up_direction'] = "MATCH (Y-down)"
    else:
        comparison['issues'].append(
            f"Up direction mismatch: Human Y-comp={h_up['y_component']:.3f}, "
            f"Mouse Y-comp={m_up['y_component']:.3f}"
        )
        comparison['recommendations'].append(
            "Check normalize_cameras_to_y_up() target direction (+Y vs -Y)"
        )

    # Elevation comparison
    h_elev = human['stats']['elevation_mean']
    m_elev = mouse['stats']['elevation_mean']

    comparison['details']['elevation'] = {
        'human_mean': h_elev,
        'human_std': human['stats']['elevation_std'],
        'mouse_mean': m_elev,
        'mouse_std': mouse['stats']['elevation_std'],
    }

    if abs(h_elev - m_elev) < 5:
        comparison['summary']['elevation'] = f"SIMILAR (H={h_elev:.1f}°, M={m_elev:.1f}°)"
    else:
        comparison['issues'].append(
            f"Elevation difference: Human={h_elev:.1f}°, Mouse={m_elev:.1f}°"
        )

    # Planarity check
    if 'orbit_analysis' in human and 'orbit_analysis' in mouse:
        h_plan = human['orbit_analysis']['planarity']
        m_plan = mouse['orbit_analysis']['planarity']

        comparison['details']['orbit'] = {
            'human_planarity': h_plan,
            'mouse_planarity': m_plan,
            'human_normal': human['orbit_analysis']['orbit_normal'],
            'mouse_normal': mouse['orbit_analysis']['orbit_normal'],
        }

    return comparison


def generate_markdown_report(human: dict, mouse: dict, comparison: dict, output_path: str):
    """Generate detailed markdown report."""

    report = f"""# Camera Specification Comparison Report

**생성일**: {comparison['timestamp']}
**목적**: 인간 vs 생쥐 데이터 카메라 파라미터 정량적 비교

---

## 1. 요약 (Summary)

| 항목 | 상태 | 인간 | 생쥐 |
|------|------|------|------|
| 이미지 크기 | {comparison['summary'].get('image_size', 'N/A')} | {human['image_size']} | {mouse['image_size']} |
| 카메라 거리 | {comparison['summary'].get('camera_distance', 'N/A')} | {human['stats']['distance_mean']:.3f} | {mouse['stats']['distance_mean']:.3f} |
| Up 방향 | {comparison['summary'].get('up_direction', 'N/A')} | Y-comp: {human['up_analysis']['y_component']:.3f} | Y-comp: {mouse['up_analysis']['y_component']:.3f} |
| Focal Length | {comparison['summary'].get('focal_length', 'N/A')} | {human['intrinsics']['fx']:.2f} | {mouse['intrinsics']['fx']:.2f} |
| Elevation | {comparison['summary'].get('elevation', 'N/A')} | {human['stats']['elevation_mean']:.1f}° | {mouse['stats']['elevation_mean']:.1f}° |

---

## 2. 발견된 이슈

"""

    if comparison['issues']:
        for issue in comparison['issues']:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "✅ 주요 이슈 없음\n"

    report += """
---

## 3. 권장 조치

"""

    if comparison['recommendations']:
        for rec in comparison['recommendations']:
            report += f"- 📋 {rec}\n"
    else:
        report += "✅ 추가 조치 불필요\n"

    report += f"""
---

## 4. 상세 카메라 분석

### 4.1 인간 데이터 ({human['name']})

| 카메라 | 거리 | Elevation | Azimuth | Up Vector |
|--------|------|-----------|---------|-----------|
"""

    for i, cam in enumerate(human['cameras']):
        up = cam['up_vector']
        report += f"| cam_{i:03d} | {cam['distance']:.3f} | {cam['elevation']:.1f}° | {cam['azimuth']:.1f}° | [{up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f}] |\n"

    report += f"""
**통계**:
- 거리: mean={human['stats']['distance_mean']:.3f}, std={human['stats']['distance_std']:.4f}
- Elevation: mean={human['stats']['elevation_mean']:.1f}°, range=[{human['stats']['elevation_min']:.1f}°, {human['stats']['elevation_max']:.1f}°]

### 4.2 생쥐 데이터 ({mouse['name']})

| 카메라 | 거리 | Elevation | Azimuth | Up Vector |
|--------|------|-----------|---------|-----------|
"""

    for i, cam in enumerate(mouse['cameras']):
        up = cam['up_vector']
        report += f"| cam_{i:03d} | {cam['distance']:.3f} | {cam['elevation']:.1f}° | {cam['azimuth']:.1f}° | [{up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f}] |\n"

    report += f"""
**통계**:
- 거리: mean={mouse['stats']['distance_mean']:.3f}, std={mouse['stats']['distance_std']:.4f}
- Elevation: mean={mouse['stats']['elevation_mean']:.1f}°, range=[{mouse['stats']['elevation_min']:.1f}°, {mouse['stats']['elevation_max']:.1f}°]

---

## 5. Up Direction 분석

| 데이터셋 | 평균 Up Vector | Y 성분 | 정렬 상태 |
|----------|----------------|--------|-----------|
| 인간 | {human['up_analysis']['avg_up_vector']} | {human['up_analysis']['y_component']:.3f} | {'Y-up' if human['up_analysis']['is_y_up'] else 'Y-down' if human['up_analysis']['is_y_down'] else 'Mixed'} |
| 생쥐 | {mouse['up_analysis']['avg_up_vector']} | {mouse['up_analysis']['y_component']:.3f} | {'Y-up' if mouse['up_analysis']['is_y_up'] else 'Y-down' if mouse['up_analysis']['is_y_down'] else 'Mixed'} |

"""

    if 'orbit_analysis' in human and 'orbit_analysis' in mouse:
        report += f"""
---

## 6. Orbit Plane 분석

| 데이터셋 | Planarity | Orbit Normal |
|----------|-----------|--------------|
| 인간 | {human['orbit_analysis']['planarity']:.4f} | {human['orbit_analysis']['orbit_normal']} |
| 생쥐 | {mouse['orbit_analysis']['planarity']:.4f} | {mouse['orbit_analysis']['orbit_normal']} |

*Planarity: 1.0 = 완벽한 평면, 카메라들이 하나의 평면에 배치됨*
"""

    report += f"""
---

## 7. 결론

### 핵심 발견사항

"""

    # Key findings based on analysis
    if comparison['issues']:
        report += "**⚠️ 카메라 정규화 문제 발견:**\n\n"
        for issue in comparison['issues']:
            report += f"1. {issue}\n"
    else:
        report += "**✅ 카메라 정규화 정상:**\n\n"
        report += "- 인간 데이터와 생쥐 데이터의 카메라 스펙이 일치합니다.\n"
        report += "- GS-LRM 학습 시 동일한 입력 형식으로 처리됩니다.\n"

    report += """
### 흐릿한 출력 원인 가능성

1. **카메라 Up Direction 불일치** (가능성 높음)
   - pose-splatter: target_up = [0, -1, 0] (Y-down)
   - mouse_dataset: target_up = [0, +1, 0] (Y-up)
   - GS-LRM pretrained 모델이 기대하는 convention 확인 필요

2. **카메라 거리 정규화 미적용 또는 오적용**
   - 정규화 후 거리가 2.7에 가까운지 확인

3. **Elevation 차이**
   - 인간 데이터와 생쥐 데이터의 elevation 분포가 다를 수 있음

---

*이 보고서는 자동 생성되었습니다.*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved: {output_path}")


def main():
    # Paths
    facelift_root = Path(__file__).parent.parent

    # Human sample data path
    human_camera_path = facelift_root / "data_sample" / "gslrm" / "sample_000" / "opencv_cameras.json"

    # Alternative: use utils_folder reference cameras
    utils_camera_path = facelift_root / "utils_folder" / "opencv_cameras.json"

    # Mouse sample data path (use first available)
    mouse_data_dir = facelift_root / "data_mouse"
    mouse_train_list = mouse_data_dir / "data_mouse_train.txt"

    # Output paths
    output_dir = facelift_root / "outputs" / "camera_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = facelift_root / "docs" / "reports" / "251218_camera_spec_comparison_report.md"

    print("=" * 60)
    print("Camera Specification Comparison: Human vs Mouse")
    print("=" * 60)

    # Load human data
    print("\n[1] Loading human camera data...")
    if human_camera_path.exists():
        human_data = load_opencv_cameras(str(human_camera_path))
        human_name = "data_sample/gslrm/sample_000"
        print(f"  Loaded: {human_camera_path}")
    elif utils_camera_path.exists():
        human_data = load_opencv_cameras(str(utils_camera_path))
        human_name = "utils_folder (inference reference)"
        print(f"  Loaded: {utils_camera_path}")
    else:
        print("  ERROR: No human camera data found!")
        return

    # Load mouse data
    print("\n[2] Loading mouse camera data...")
    if mouse_train_list.exists():
        with open(mouse_train_list, 'r') as f:
            mouse_samples = f.read().strip().split('\n')

        # Use first sample
        mouse_sample_path = Path(mouse_samples[0].strip())
        mouse_camera_path = mouse_sample_path / "opencv_cameras.json"

        if mouse_camera_path.exists():
            mouse_data = load_opencv_cameras(str(mouse_camera_path))
            mouse_name = str(mouse_sample_path.name)
            print(f"  Loaded: {mouse_camera_path}")
        else:
            print(f"  ERROR: Mouse camera file not found: {mouse_camera_path}")
            return
    else:
        print(f"  ERROR: Mouse train list not found: {mouse_train_list}")
        return

    # Analyze cameras
    print("\n[3] Analyzing camera parameters...")
    human_analysis = analyze_cameras(human_data['frames'], human_name)
    mouse_analysis = analyze_cameras(mouse_data['frames'], mouse_name)

    print(f"  Human: {human_analysis['num_views']} views, distance={human_analysis['stats']['distance_mean']:.3f}")
    print(f"  Mouse: {mouse_analysis['num_views']} views, distance={mouse_analysis['stats']['distance_mean']:.3f}")

    # Compare
    print("\n[4] Comparing specifications...")
    comparison = compare_results(human_analysis, mouse_analysis)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for key, value in comparison['summary'].items():
        print(f"  {key}: {value}")

    if comparison['issues']:
        print("\n⚠️ ISSUES FOUND:")
        for issue in comparison['issues']:
            print(f"  - {issue}")
    else:
        print("\n✅ No major issues found")

    # Generate report
    print("\n[5] Generating report...")
    generate_markdown_report(human_analysis, mouse_analysis, comparison, str(report_path))

    # Save JSON results
    results_json = {
        'human': human_analysis,
        'mouse': mouse_analysis,
        'comparison': comparison,
    }

    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"JSON results saved: {json_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
