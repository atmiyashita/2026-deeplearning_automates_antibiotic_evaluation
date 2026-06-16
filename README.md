# Silkworm Monitoring Scripts (Release Bundle, 2026-06-16)

This bundle is prepared for publishing to Miyashita lab GitHub.

## Included

- `detection/`
  - `train_kaiko_fasterrcnn_2025_0219.py` (2025-0219モデル学習時のコード本体。パスのみプレースホルダ化)
  - `infer_kaiko_fasterrcnn_public.py` (公開用推論コード。入力フォルダ参照はプレースホルダ)
  - `requirements.txt` (PyTorch-side minimum dependencies)
- `acquisition/`
  - `app_qt_v2.py` (Raspberry Pi camera capture/recording GUI)
  - `requirements.txt`
- `docs/source_manifest.tsv`
  - Mapping from bundled files to original source paths.

## Quick start (detection)

```bash
cd detection
pip install -r requirements.txt

# 1) train_kaiko_fasterrcnn_2025_0219.py の path1..path9 を実データに置換
python train_kaiko_fasterrcnn_2025_0219.py

# 2) infer_kaiko_fasterrcnn_public.py の
#    checkpoint_path / image_folder を実パスに置換
python infer_kaiko_fasterrcnn_public.py
```

## Notes

- Datasets, model weights, and output files are **not** included in this bundle.
- Model weights (`.pth`) are intentionally not included for IP protection.
- `acquisition/app_qt_v2.py` is intended for Raspberry Pi environments with camera modules.
- `acquisition/app_qt_v2.py` was copied from:
  `https://github.com/miyauchi1224824/automated_observation_and_detection` (retrieved on 2026-06-16).
- Please confirm licensing/attribution policy before public redistribution.
- Please add an explicit repository LICENSE before public release.
