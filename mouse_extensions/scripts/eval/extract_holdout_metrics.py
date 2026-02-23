#!/usr/bin/env python3
"""
Extract holdout-specific metrics from existing FL fair eval JSONs.

For each view condition (6v/5v/4v), extract the holdout view's metrics
from the corresponding GS-LRM model and E2E model fair eval results.

Output: A unified JSON with all 9 experiments' metrics organized by
view condition and evaluation protocol.

Usage:
    python extract_holdout_metrics.py --fair-dir experiments/comparison/tier/ \
        --ps-json /tmp/ps_m5_fair_results.json \
        --output experiments/comparison/9exp_unified_metrics.json
"""

import argparse
import json
from pathlib import Path


# View condition definitions
VIEW_CONDITIONS = {
    "6v": {
        "total_views": 6,
        "train_views": [0, 1, 2, 3, 4],
        "holdout_view": 5,
        "holdout_key": "view_5",
        "fl_gslrm_model": "5view",  # 5-view GS-LRM for 6v condition
        "fl_gslrm_json": "gslrm_5view_fair.json",
        "ps_experiment": "m5_baseline_gs",
    },
    "5v": {
        "total_views": 5,
        "train_views": [0, 1, 2, 3],
        "holdout_view": 4,
        "holdout_key": "view_4",
        "fl_gslrm_model": "4view",
        "fl_gslrm_json": "gslrm_4view_fair.json",
        "ps_experiment": "m5_5view_holdout4",
    },
    "4v": {
        "total_views": 4,
        "train_views": [0, 1, 2],
        "holdout_view": 3,
        "holdout_key": "view_3",
        "fl_gslrm_model": "3view",
        "fl_gslrm_json": "gslrm_3view_fair.json",
        "ps_experiment": "m5_4view_holdout3",
    },
}

METRICS = [
    "psnr_gt_masked",
    "psnr_intersection",
    "iou",
    "coverage",
    "ssim_gt_masked",
]

E2E_JSON = "e2_resume_20k_fair.json"


def extract_metrics(data: dict, view_key: str = None) -> dict:
    """Extract metric means from a fair eval JSON.

    Args:
        data: Fair eval JSON data
        view_key: If provided, extract from per_view[view_key].
                  If None, extract from overall.
    """
    if view_key:
        source = data.get("per_view", {}).get(view_key, {})
    else:
        source = data.get("overall", {})

    result = {}
    for m in METRICS:
        if m in source:
            val = source[m]
            if isinstance(val, dict):
                result[m] = {
                    "mean": val.get("mean"),
                    "std": val.get("std"),
                    "n": val.get("n"),
                }
            else:
                result[m] = {"mean": val, "std": None, "n": None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-dir", type=str, required=True,
                       help="Directory with FL fair eval JSONs")
    parser.add_argument("--ps-6v-json", type=str, default=None,
                       help="PS M5 6v fair eval JSON")
    parser.add_argument("--ps-5v-json", type=str, default=None,
                       help="PS M5 5v fair eval JSON")
    parser.add_argument("--ps-4v-json", type=str, default=None,
                       help="PS M5 4v fair eval JSON")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    fair_dir = Path(args.fair_dir)
    result = {
        "description": "9-experiment unified metrics for FL vs PS comparison",
        "evaluation_protocol": "evaluation_protocol_v1.0",
        "view_conditions": {},
    }

    # Load E2E data once
    e2e_path = fair_dir / E2E_JSON
    e2e_data = None
    if e2e_path.exists():
        e2e_data = json.load(open(e2e_path))
        print(f"Loaded E2E: {e2e_path}")

    ps_jsons = {
        "6v": args.ps_6v_json,
        "5v": args.ps_5v_json,
        "4v": args.ps_4v_json,
    }

    for vc_name, vc in VIEW_CONDITIONS.items():
        print(f"\n=== View Condition: {vc_name} ===")
        vc_result = {
            "config": {
                "total_views": vc["total_views"],
                "train_views": vc["train_views"],
                "holdout_view": vc["holdout_view"],
            },
            "experiments": {},
        }

        # --- Protocol A: Temporal (all views, test frames) ---
        # FL GS-LRM
        gslrm_path = fair_dir / vc["fl_gslrm_json"]
        if gslrm_path.exists():
            gslrm_data = json.load(open(gslrm_path))
            vc_result["experiments"]["fl_gslrm"] = {
                "name": f"FL GS-LRM {vc['fl_gslrm_model']}",
                "method": "facelift",
                "pipeline": "gslrm",
                "input_views": len(vc["train_views"]),
                "protocol_a_temporal": extract_metrics(gslrm_data),
                "protocol_b_spatial": extract_metrics(
                    gslrm_data, vc["holdout_key"]),
            }
            psnr_a = vc_result["experiments"]["fl_gslrm"][
                "protocol_a_temporal"].get("psnr_gt_masked", {}).get("mean")
            psnr_b = vc_result["experiments"]["fl_gslrm"][
                "protocol_b_spatial"].get("psnr_gt_masked", {}).get("mean")
            print(f"  FL GS-LRM: A={psnr_a:.2f}, B(holdout {vc['holdout_key']})={psnr_b:.2f}" if psnr_a and psnr_b else f"  FL GS-LRM: loaded")
        else:
            print(f"  WARNING: {gslrm_path} not found")

        # FL E2E
        if e2e_data:
            vc_result["experiments"]["fl_e2e"] = {
                "name": "FL E2E (1-view input)",
                "method": "facelift",
                "pipeline": "e2e",
                "input_views": 1,
                "note": "Same model for all view conditions, only eval view changes",
                "protocol_a_temporal": extract_metrics(e2e_data),
                "protocol_b_spatial": extract_metrics(
                    e2e_data, vc["holdout_key"]),
            }
            psnr_a = vc_result["experiments"]["fl_e2e"][
                "protocol_a_temporal"].get("psnr_gt_masked", {}).get("mean")
            psnr_b = vc_result["experiments"]["fl_e2e"][
                "protocol_b_spatial"].get("psnr_gt_masked", {}).get("mean")
            print(f"  FL E2E: A={psnr_a:.2f}, B(holdout {vc['holdout_key']})={psnr_b:.2f}" if psnr_a and psnr_b else f"  FL E2E: loaded")

        # PS M5
        ps_json_path = ps_jsons.get(vc_name)
        if ps_json_path and Path(ps_json_path).exists():
            ps_data = json.load(open(ps_json_path))
            vc_result["experiments"]["ps_m5"] = {
                "name": f"PS M5 {vc_name}",
                "method": "pose-splatter",
                "pipeline": "per-scene",
                "input_views": len(vc["train_views"]),
                "protocol_a_temporal": extract_metrics(ps_data),
                "protocol_b_spatial": extract_metrics(
                    ps_data, vc["holdout_key"]),
            }
            psnr_a = vc_result["experiments"]["ps_m5"][
                "protocol_a_temporal"].get("psnr_gt_masked", {}).get("mean")
            print(f"  PS M5: A={psnr_a:.2f}" if psnr_a else "  PS M5: loaded")
        else:
            vc_result["experiments"]["ps_m5"] = {
                "name": f"PS M5 {vc_name}",
                "status": "pending_training",
                "experiment": vc["ps_experiment"],
            }
            print(f"  PS M5: PENDING (need training)")

        result["view_conditions"][vc_name] = vc_result

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Protocol A (Temporal) - PSNR_gt_masked")
    print("=" * 80)
    print(f"{'Condition':<10} {'FL GS-LRM':>12} {'FL E2E':>12} {'PS M5':>12}")
    print("-" * 46)
    for vc_name in ["6v", "5v", "4v"]:
        vc = result["view_conditions"][vc_name]
        vals = []
        for exp in ["fl_gslrm", "fl_e2e", "ps_m5"]:
            d = vc["experiments"].get(exp, {})
            pa = d.get("protocol_a_temporal", {})
            v = pa.get("psnr_gt_masked", {}).get("mean")
            vals.append(f"{v:.2f}" if v else "---")
        print(f"{vc_name:<10} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("\n" + "=" * 80)
    print("SUMMARY: Protocol B (Spatial NVS) - PSNR_gt_masked (holdout view)")
    print("=" * 80)
    print(f"{'Condition':<10} {'Holdout':>8} {'FL GS-LRM':>12} {'FL E2E':>12} {'PS M5':>12}")
    print("-" * 54)
    for vc_name in ["6v", "5v", "4v"]:
        vc = result["view_conditions"][vc_name]
        hv = VIEW_CONDITIONS[vc_name]["holdout_key"]
        vals = []
        for exp in ["fl_gslrm", "fl_e2e", "ps_m5"]:
            d = vc["experiments"].get(exp, {})
            pb = d.get("protocol_b_spatial", {})
            v = pb.get("psnr_gt_masked", {}).get("mean")
            vals.append(f"{v:.2f}" if v else "---")
        print(f"{vc_name:<10} {hv:>8} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")


if __name__ == "__main__":
    main()
