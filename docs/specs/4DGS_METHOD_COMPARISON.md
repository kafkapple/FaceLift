# 4D Gaussian Splatting Method Comparison for FaceLift

> **Full document moved to Obsidian** (이론/분석/전략 문서 — 실행에 직접 불필요)

## Location

**Obsidian**: `30_Projects/2603_NeurIPS_3D-Animal-Recon/docs/concepts/55_4DGS_METHOD_COMPARISON.md`

## Quick Reference

- DRoPS/DeGauss/MonoFusion 3종 상세 비교 (Section 2-4)
- Comprehensive comparison tables (Section 5)
- Hybrid "Deform V4" long-term architecture (Section 6)
- Verification strategy (Section 7)
- **Sequential Pilot 실행 전략** (Section 9) — MonoFusion pilot → Gate → Commit/Fallback

## Key Conclusions

1. **MonoFusion** (5/5): Canonical + static appearance = identity, SE(3) = behavior feature
2. **DRoPS** (3/5): Isometry loss만 차용 가능 (pre-scan 비호환, 코드 미공개)
3. **DeGauss** (1/5): Gaussian tracking 없음 — 기존 5/5 → 1/5 재평가
4. **전략**: Sequential pilot (MonoFusion 2주 → Gate → Commit or GS-LRM+DRoPS fallback)

---

*Pointer file | v2.0 | 2026-03-31*
