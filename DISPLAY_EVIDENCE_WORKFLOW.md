# 结构化诊断、凹处选位与 Photoshop 执行

试验入口为 `workflow=evidence_v1`。普通批量使用 PNG，独立 Photoshop 入口读取同一合成流程导出的图层清单。两者共用视觉证据和执行计划协议。

## 运行方式

沿用项目 Python 环境。把 `workflow` 设为 `evidence_v1` 可从 CLI、API 或网页启动该流程。实物尺度复验使用 `geometry_preset=rebuild` 和带来源的 `vehicle_reference`。填写 `prepare_photoshop=true` 时，程序为通过硬性检查的合成生成 `layers-N/composition.json`。已冻结的 12 项小样有独立清单、原始尝试和后续账本。

```powershell
work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/batch_display.py run --config <配置.json> --workflow evidence_v1
work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/refine_photoshop.py --manifest <composition.json> --out <新的输出目录> --iterations 2
```

运行 Photoshop 入口需要本机已注册的 Photoshop 与 MCP。程序可通过 COM 启动已安装的 Photoshop。新的 PSD 从原对象图层和固定背景图层重建，使用剪贴调整层执行亮度、色偏和饱和度。每轮以实际 PNG 复评，保存 PSD 并重开核对。采样、雾化、局部对比与单次预乘透明度的相机过滤由共享合成步骤完成。PSD 展示透明度用于复现显示颜色，原几何透明度及光学覆盖另存到绑定清单。

视觉诊断输出 `visual-evidence-v1`，绑定当前完整图的散列和目标 ID。执行模型返回 `execution-plan-v1` 绝对值计划，程序审核证据、值域、执行方向、重复计划和阶段依赖。视觉和执行模型读取独立服务配置中的 `vision`、`planner`，用量按实际请求记账。交互式 Token Plan 可在既有看图命令中增加 `--format visual_evidence`。

## 选位与参数库

先在整图核对车厢内沿和横截面宽度对应。有对应证据才进入完整场景分割和投影。完整选位在多尺度相对深度上区分槽底、挡边、凸起和空隙，再按真实轮廓、接触簇、净空和保护区筛选。估计相机下的投影适配与现场接触证据单独保存。

参数数据库位于 `work/knowledge/history-v2.sqlite`。新 PNG 批次完成时索引结果和请求账本。试验将只读快照与输入散列冻结，后续导入写入主库。规则按材质、投影尺寸及局部光色、雾、清晰度、噪声匹配，算法版本也必须一致。归纳先按独立背景组聚合，至少三个拟合组和两个留出组才可验证规则。留出摄影分数需改善，结构和像素检查需通过。缺少已验证规则时使用保守初始值。

```powershell
$env:PYTHONPATH = 'src'
work/triposr-env/Scripts/python.exe -X utf8 -B -m scrap_composite.knowledge_cli --db work/knowledge/history-v2.sqlite import --history-root <历史结果目录>
work/triposr-env/Scripts/python.exe -X utf8 -B -m scrap_composite.knowledge_cli --db work/knowledge/history-v2.sqlite derive
work/triposr-env/Scripts/python.exe -X utf8 -B -m scrap_composite.knowledge_cli --db work/knowledge/history-v2.sqlite report --out work/knowledge/report-v2
```

维护代码时使用 `src/scrap_composite` 中的共享协议、知识库、凹处算法和 Photoshop 适配器。现有 Blender、队列、CLI 和网页逐步委托共享模块。核心包的依赖和固定 JSX 纳入 `pyproject.toml`，实现指纹覆盖共享源码。复跑前若实现指纹变化，可用同一冻结清单和新尝试目录记录运行。

## 实际验证

| 项目 | 结果 |
|---|---|
| 历史凹处诊断 | 16 张完整画面均找到图像候选，14 张至少有一组能容纳旧姿态真实轮廓和接触斑，耗时 38.14 秒。相机沿用历史估计，现场接触待审。 |
| Photoshop 夹具 | 本机 Photoshop 26.6.0 实际执行三项调整。16 项验证通过，PSD 重开与 PNG 逐像素一致，耗时 11.641 秒。 |
| 结构化视觉协议 | 归档图由 `qwen3.7-plus` 返回符合新版协议的证据，1 次请求，4,012 token。 |
| 12 项实景留出 | 03、13、31、33 × 相机 36 的三张背景。三张都无可核对的内沿横截面宽度对应，12 项停止在尺度预检，0 张 PNG、0 个 PSD，摄影验收 0/12。44 个源模型散列一致。 |
| 参数库 | 248 条历史记录加 25 条本轮尝试，共 273 条。没有足够的跨背景配对证据，批准规则为 0。 |

首轮服务连接中断。第二次尝试暴露了完整场景 JSON 的多边形格式不稳定，以及同一场景失败被多个物品重复请求的问题。最终把宽度证据检查提前，并在同一批次复用场景失败。最终尝试用 3 次请求完成 12 项判定，耗时 5.813 秒，渲染与落稳调用为 0。所有尝试仍在同一账本范围内。

实景小样累计 25 次请求，73,069 token。已报告用量部分的 API 估算下界为 ¥0.11522372，1 次中断请求未报告用量。另有交互诊断 1 次、4,012 token，Credits 未返回。本轮合计 26 次请求、77,081 已报告 token，总现金费用未知。

[留出小样完整画面与成本](outputs/结构化执行留出小样-20261112/index.html) · [历史凹处对照](outputs/凹处支持证据-历史16-20261112/index.html) · [知识库报告](work/knowledge/report-v2/index.html) · [真实 Photoshop 夹具](work/photoshop-qa/adapter-rgb-20261112/export/result.json)

实景验收仍缺可见的车厢宽度对应或经核实的场景标尺。通过这一步后，才能核对新摆位的实际接触、自然度和 Photoshop 复评。
