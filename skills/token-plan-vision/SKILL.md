---
name: token-plan-vision
description: Use Alibaba Token Plan qwen3.7-plus for image diagnosis explicitly requested in the current Codex conversation. Useful for material, lighting, scale and contact feedback on scrap composites; not an unattended batch provider.
---

# Token Plan interactive vision

Run one bounded visual question through the project's `interactive_token_plan.py`.
Use 1–4 explicit existing images from the current question, with a UTF-8 question file.
The tool returns structured Chinese evidence and provider token usage. It does not render,
edit images, place objects, resume batches or mark production output accepted.

Project: `D:/CodexProjects/experiments/scrap-composite`.
From that directory, run:

```powershell
work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/interactive_token_plan.py --image <full-image> --question-file <question.txt> --out <new-output-directory>
```

Use `--format visual_evidence` for the versioned seven-score protocol. In that
mode the first image is the current frame and any remaining images are context.
The report binds findings and normalized regions to the first image's hash.
For industrial documentary photography, use `--format photographer` with a short
question of up to 240 characters. It returns the same `photo-brief-v1` factors as
the local photography workflow. See [scrap-photography](../../.agents/skills/scrap-photography/SKILL.md).

The default is `qwen3.7-plus` on the official Beijing Token Plan endpoint. Credentials
come from `BAILIAN_TOKEN_PLAN_API_KEY` or the application's current-user DPAPI store;
never include a key in the question, command, repository or output.

Preserve full-frame image links for the user. When checking support, explain that a
virtual receiver or plausible shadow does not establish contact with the real scene.
Keep uncertain dimensions unknown. Summarize evidence and do not silently apply model
suggestions. User pause remains in force. Exact-input output directories are reused
without requests; failures require review, not an automatic new loop.

This skill is for calls initiated interactively inside Codex. Do not connect it to the
batch queue, API backend or a scheduled loop. This is the provider's usage boundary:
[Token Plan personal scope](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview)
and [official vision integration](https://help.aliyun.com/zh/model-studio/add-vision-skill).
The existing independent batch-provider configuration remains separate.

Report actual requests, retries and reported tokens. Credits/cash cost remain unknown
unless provided by the subscription console; do not reuse DeepSeek API tariffs.
