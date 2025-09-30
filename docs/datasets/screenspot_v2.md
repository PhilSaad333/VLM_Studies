# ScreenSpot-v2 Benchmark Notes

Dataset: `HongxinLi/ScreenSpot_v2`

Each example provides:
- `image`: 960×540 screenshot (PIL image)
- `bbox`: normalized `[x1, y1, x2, y2]` coordinates of the target region
- `instruction`: textual description (e.g., "close this window")
- `data_type`: category (icon, text, etc.)
- `data_source`: dataset origin (windows, mac, etc.)

Use `data_sources.screenspot.load_screenspot(split)` to iterate examples. The returned `ScreenSpotExample` offers:
- `bbox_pixels()` -> `(x1, y1, x2, y2)` in pixel space
- `draw_bbox(color)` -> image with rectangle overlay

This benchmark evaluates GUI perception; we will use it both for zero-shot checks and post-training evaluation.
