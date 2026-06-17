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
- `analysis/`
  - `pava_survival_plot_public.py` (PAVA-based monotonic post-processing and survival-curve plotting from CSV input)
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

# 3) PAVA補正と生存曲線描画
cd ../analysis
pip install -r requirements.txt
python pava_survival_plot_public.py \
  --input-csv survival_input.csv \
  --time-col elapsed_hours \
  --survival-cols PAO1,PAO1_meropenem,saline \
  --output-dir outputs/survival_plot
```

## Notes

- Datasets, model weights, and output files are **not** included in this bundle.
- `acquisition/app_qt_v2.py` is intended for Raspberry Pi environments with camera modules.
- `acquisition/app_qt_v2.py` was copied from:
  `https://github.com/miyauchi1224824/automated_observation_and_detection` (retrieved on 2026-06-16).
- Please confirm licensing/attribution policy before public redistribution.
- Please add an explicit repository LICENSE before public release.
