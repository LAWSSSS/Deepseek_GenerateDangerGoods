---
name: industrial-photo-review
description: Diagnose and execute evidence-bound photographic and PBR corrections for the scrap-composite project using its typed observer/editor and Photoshop readback. Use for model realism, industrial-scene integration, or targeted material and lighting repair; preserve source geometry and separate scene evidence from studio identity references.
---

# Industrial model photography

Use the project's current public package and `tools/ps-model-controller` entrypoints. Read `README.md`, `AGENTS.md` and the current run manifest before choosing a backend. Do not infer a current control value from a previous screenshot.

## Diagnose by cause

Keep image roles explicit: the current complete frame is the evaluation target; its local scene supports photographic comparisons; source photos support identity and parts; gray previews isolate geometry. A studio reference does not establish scene exposure, haze or sharpness.

For each defect record the observed fact separately from its cause hypothesis, its counterevidence and the measurement needed to distinguish causes. Read [photographic reasoning](references/photographic-reasoning.md) for material/light ambiguity. Where the question concerns one component, request its native-resolution view as well as the full frame rather than infer microdetail from a thumbnail.

Use the project's typed photographic protocol. Bind findings to image hash, target/part, region and reference role. Bind execution to that diagnosis and the current parameter/state hash. Read available operations and their units before proposing a change. Unknown or contradictory evidence is not a successful inspection.

## Select the responsible executor

- Blender: roughness distribution, oxide/scratch coverage, physically scaled relief, surface deposits, broad light and fill. Geometry errors return to asset/contact validation.
- Photoshop: read document/layer identity, profile, actual curves, saturation, masks and clipping; apply only registered absolute adjustments on a copy.
- Existing imaging module: atmosphere, alpha-aware camera blur and noise, within one camera budget. Do not add another fog or blur layer in Photoshop.

Photoshop curve offsets are code values, not exposure EV. Channel curves are not measured Kelvin. Pixel statistics are not BRDF or incident radiance. Retain the origin, sampling region and confidence of each measurement.

## Execute and verify

Follow inspect → measure → diagnose → plan → execute → read back → evaluate actual export. Use document/layer IDs and an unchanged state fingerprint. Never send model-authored JSX or Python to an unrestricted executor. Preserve original pixels outside the allowed masks, alpha holes, foreground and source assets.

Use at most three related operations for one cause per round; the configured iteration limit is an upper bound. Reject repeated, out-of-range, reversed or ineffective operations before another API call. Verify effective values and pixels; do not transfer previous-image scores. Retain the better program-valid version and report unsupported causes.

Use the normal concise diagnostic on known simple corrections. Use the detailed modules for first asset review, mixed material, unresolved cause, contradictions or stalled progress. Summaries may be short; evidence, measurement units and relevant protected parts must not be dropped to save tokens.

## This project's model checkpoint

For the four-asset realism experiment, deliver the gray/source/new material, relighting and turning plates first. Scene rendering waits for the user's model checkpoint. No asset pass implies scene scale or contact passed. Per-asset material variants are allowed; background-specific albedo or wear recipes are not.

Token Plan is invoked only for a bounded question initiated in the current Codex conversation, using the separate `token-plan-vision` entrypoint. Do not connect it to an unattended queue. Record requests, retries, tokens and unknown cash/credit prices separately from rendering.

## Provenance

Adapted workflow ideas from [photo-editing-craft](https://github.com/aipiratesMunich/photo-editing-craft/tree/448f34fc8e53f406e8c002cf061c0471fe6030c2), pinned commit `448f34fc8e53f406e8c002cf061c0471fe6030c2`; reviewed `SKILL.md`, `craft/compositing.md` and `craft/workflow-order.md`. The upstream MIT notice is in `LICENSE.upstream`. Its fixed saturation, haze, edge feathering and hand-painted shadow recipes are not this project's acceptance criteria.
