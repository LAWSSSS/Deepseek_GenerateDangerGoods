---
name: scrap-photography
description: Review and refine scrap-yard composites as an industrial documentary photographer. Use for local lighting, reflections, wear, haze, tonal range, color, camera texture, and structured handoff to the bounded retouching planner.
---

# 废钢纪实摄影复核

以现场监控相机为成像参照。首图是待评审的完整画面，局部图帮助辨认细节，资产参考图用于确认材质与标识。对照附近同层、同朝向、同类表面的线索。把接触、光照、材质、空气与相机因素分别判断。

- 视觉模型使用 [摄影师观察指令](../../../src/scrap_composite/photo_observer.txt)。记录当前全图坐标、可见线索、参照、方向和可信度。按因素使用协议词表。评分低且缺少可见线索时保留不确定。
- 执行模型使用 [摄影师修图指令](../../../src/scrap_composite/photo_editor.txt)。从绑定当前成图和证据的选项中选择一项，或保留待审。修正后用实际导出的 PNG 复核；保持物品身份、孔洞、前景及保护像素。
- 复核已存在的全景使用 `tools/ps-model-controller/photo_review.py review`。有 `composition.json` 时可预览 Photoshop 计划。正常合成或 Photoshop 入口使用 `--review-profile photographer`。

遇到受光、磨损、雾幕或相机细节相互混淆时，读 [摄影判断与操作映射](references/photographic-factors.md)。复用原始图和程序检查，按实际请求与 token 记账。使用用户授权的模型服务。
