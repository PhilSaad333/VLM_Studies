# AGUVIS Dataset Notes

`smolagents/aguvis-stage-1` and `smolagents/aguvis-stage-2` provide GUI-grounding supervision. Each subset (config) returns records with `images`, `texts`, and `source` fields.

## Stage-1 Overview
- Configs: `guienv`, `omniact`, `ricoig16k`, `ricosca`, `seeclick`, `ui_refexp`, `webui350k`, `widget_captioning`.
- Each example typically contains one 1080p screenshot (PIL image).
- `texts[0]` holds a dictionary `{'user': instruction, 'assistant': action}` where `assistant` is usually a single function call (e.g. `click(x=0.5, y=0.6963)`).
- Coordinates are normalized floats in `[0,1]`.

### Normalized Loader
`data_sources.aguvis.stage1.load_stage1(config)` yields dictionaries with keys `image`, `user`, `assistant`, `source`.

## Stage-2 Overview
- Configs include `mind2web-*`, `miniwob-*`, `aitw-*`, `guiact-*`, etc. (20 total).
- Examples feature a screenshot and a chat-style record with `system`, `user`, and `assistant` fields.
- `assistant` strings interleave `<think>` reasoning with `<code>` tool invocations (`click`, `scroll`, `type`, etc.).

### Normalized Loader
`data_sources.aguvis.stage2.load_stage2(config)` yields dictionaries with keys `image`, `system`, `user`, `assistant`, `source`.

## Notes
- Both stages streamable via `load_dataset(..., streaming=True)`; images arrive as `PIL.Image` objects.
- Use `source` to track the original dataset for metrics per domain.
- Stage-2 instructions often list available tool functions; normalized loader preserves them in the `system` field.
