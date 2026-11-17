# 摄影师诊断与执行

`review_profile=photographer` 启用工业纪实摄影师的视觉复核和数字修图决策。两阶段由 [Pydantic Graph](https://pydantic.dev/docs/ai/graph/graph/) 编排，使用现有 `vision`、`planner` 服务配置、请求缓存和用量账本。详细判断依据在 [摄影技能](.agents/skills/scrap-photography/SKILL.md)，运行时使用其中对应的简短角色指令与因素词表。

视觉模型以完整成图为准，局部和资产参考用于辅助辨认。它分别判断主光和补光、镜面高光、材质磨损、附着灰、雾幕、曝光、黑位、高光剪裁、冷暖、绿洋红、色度、清晰度、噪声、压缩和接触影。程序可附同层亮度分位、金属和漆面的 Lab 色度、透射率、局部光向估计及像素清晰度。读图记录含可见线索、参照、归一化区域、严重度、置信度与物理量变化方向。

执行模型只接收一条优先证据、相关测量及最多三个有效绝对值选项。它选择一项或返回 `uncertain` / `tradeoff`。程序将选项、当前参数、目标与成图散列绑定。发现严重度和评分相冲突时保守降低评分；缺少参照、位置或直接几何线索的观察保留为不可判断。相反方向、明确物理问题、超出能力的主要缺陷直接结束摄影计划。光向、接收面、粗糙度、磨损与噪声仍可独立诊断，当前执行器只给出已实现且对应证据的参数。

批量合成可使用现有曝光与冷暖档、有效对比和色度、局部雾、相机锐度与表面浮灰。Photoshop 使用物品图层剪贴曲线的亮度、冷暖和饱和度。普通批量和旧记录继续使用原有入口。摄影师模式需要 `pydantic-graph==2.68.0`，本机已安装，迁移脚本也会安装。

## 入口

网页 `/batch` 的「摄影复核」选择「摄影师诊断与修正」。HTTP `/api/batches` 与 `/api/batches/test` 接受 `review_profile: "photographer"`，`GET /api/photography/schema` 返回字段、因素和预算。摄影师模式沿用 `evidence_v1` 的场景、物理和像素检查。

```powershell
$python = 'work/triposr-env/Scripts/python.exe'
$controller = 'tools/ps-model-controller'
& $python -X utf8 -B "$controller/batch_display.py" run --config 'config.json' --review-profile photographer
& $python -X utf8 -B "$controller/photo_review.py" review --image 'full.png' --out 'work/photo-review/image'
& $python -X utf8 -B "$controller/photo_review.py" review --draft 'draft-0.json' --plan --out 'work/photo-review/draft'
& $python -X utf8 -B "$controller/photo_review.py" review --manifest 'composition.json' --plan --out 'work/photo-review/layers'
& $python -X utf8 -B "$controller/refine_photoshop.py" --manifest 'composition.json' --review-profile photographer --out 'work/photo-review/export'
```

单图入口输出 `review.json` 和请求账本。`--draft` 接受当前成图的批次 `draft-N.json`，也接受历史样片中包含 `composition` 的结果。`--plan` 预览调参，`--diagnosis` 可复用同图和同一上下文的 `review.json`。`photo_review.py schema` 可在本地检查协议。已有 [Token Plan 交互技能](../../skills/token-plan-vision/SKILL.md) 可用 `--format photographer` 输出同一诊断格式。

## 验证与实际用量

116 项控制器、接口和模型回归通过。核心套件 28 项通过，真实 Photoshop 夹具跳过。摄影师模式的 Photoshop 调整、实际导出图散列交接和复评通过模拟适配器测试。项目技能校验通过。

用三张历史完整成图试跑新入口。轴承座的主要诊断为雾幕偏重，执行模型接收匹配的减雾选项后选择保留待审。液压缸的接触影与材质问题没有匹配的摄影参数。气缸有疑似接触缺陷，程序转入几何复核。调试阶段已验证一次接受减雾选项的完整计划。线上模型仍可能报告超范围区域或误判材质与接触。本轮验证的是诊断与执行交接，样片效果有待实际修正及重验。 [三张复核记录](../../outputs/摄影师双模型复核-20261113/index.html)

最终执行请求的输出为 10 token，输入为 339 token，纯文本。摄影诊断和执行模型的输出上限分别为 1,536 与 256 token。最终三图相关的有效调用为 4 次、7,974 token。全部开发与在线复验累计 22 次请求，其中 7 次重试，共 53,508 token，按本机账本估算 ¥0.0683–0.1366。具体调用和重试见 [累计账本](../../outputs/摄影师双模型复核-20261113/cost-total.json)。本次完成已有图片的诊断与计划预览，真实 Photoshop 导出和摄影自然度提升尚待验证。
