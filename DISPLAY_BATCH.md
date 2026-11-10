# 多背景批量展示合成

新增流程供人工展示与评审：分析每张背景 → 选择承托区域 → 搜索横躺姿态与朝向 → 渲染 → 同承托面匹配 → AI 复核与受限修正。直接输出原尺寸 PNG，不启动 Photoshop。

`accepted_display` 是新流程的展示状态，不改变已有 `manual_only`、模型人工确认或正式场景验收状态。已有 10 号去独立操作杆、13 号单罐、19 号拆瓶、31 号封闭缸壁修订继续由当前模型目录提供。

最新无人值守实测见[场景融合验证](#场景融合验证)。历史结果见[材质摄影和按需渲染小样验证](DISPLAY_COST_VALIDATION.md)及[摄影融合与废钢堆验证](DISPLAY_FUSION_VALIDATION.md)。

## 使用

在仓库根目录运行，先安装新增碰撞分解依赖：

```powershell
work/triposr-env/Scripts/python.exe -m pip install -r outputs/ps-model-controller/display-batch-requirements.txt
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py run --config outputs/ps-model-controller/display-batch.example.json
```

复制示例 JSON 后修改 `backgrounds` 与 `output`。默认 100 个任务、初稿之后最多修正 2 轮；`count` 允许 1–1000，`max_iterations` 允许 0–10。小于 100 的任务数用于诊断。`assets` 留空使用当前全部 44 个条目；可指定 `['3','6','10','13','19-1','31']` 对应的 JSON 字符串数组。每个任务只有一个物品。

网页和命令行示例采用 `appearance: "native"`。该模式从现场估计反射环境、接触阴影和边缘成像，增加插入痕迹判断。旧配置省略 `appearance` 时继续使用 `reference`。网页的“融合方式”可切换两者。

以下选项同样适用于旧配置。

| 字段 | 默认 | 其他值 |
|---|---|---|
| `photo_style` | `scrap_gray` 废钢灰强融合 | `retain_color` 保留更多本色 |
| `placement_mode` | `auto` 自动选表面、组合承托、局部嵌入 | `surface`、`combined`、`embedded` |
| `uncertain_policy` | `render_review` 证据不足仍出待审图 | `skip` 跳过不确定候选 |

网页沿用 `placement-web/server.py`，从工作台标题栏打开“多背景批量生成”，或访问 `/batch`。默认服务端口 8765；本轮验证实例端口为 8767。表单包含背景目录、任务数、模型列表、种子、修正轮数和设备，提供进度、结果与续跑入口。新代码需重启服务进程才能加载。

同一配置、输出目录再次运行即可续跑；已完成任务不重跑。运行中的输出目录有独占锁。输入内容、配置、算法或依赖签名改变时要求新输出目录，避免混合两版结果。恢复中的任务从可复用缓存重放已完成阶段，模型响应缓存命中不产生新请求。

## 模型服务

默认复用 `batch-config.example.json` 中的视觉服务设置。凭据只从指定环境变量、本机用户环境或既有 Harness 凭据读取；不写入配置、输出或命令参数。

其他兼容视觉服务可用 `provider_config` 指向本地 JSON：

```json
{
  "vision": {
    "url": "https://your-provider.example/v1/chat/completions",
    "model": "your-vision-model",
    "key_env": "YOUR_VISION_KEY"
  }
}
```

启动时探测真实图片输入与当前配置的结构化输出能力。`native` 的探测和成图复核使用 JSON 对象输出，旧模式默认使用 JSON function tool。图片先编码为有大小上限的 JPEG。每个逻辑请求最多两次尝试。鉴权、模型端点、视觉能力错误以及一次重试后仍不可用的限流或网络故障暂停批次。默认不设置整批 token 上限。只有配置输入、输出单价时才计算费用，否则仅报告供应商返回的 token。

百炼 `qwen3.7-plus` 可将 `provider_config` 设为 `"outputs/ps-model-controller/batch-config.qwen.example.json"`，在进程环境中配置标准百炼 API 凭据 `DASHSCOPE_API_KEY`。该配置通过模拟 HTTP 校验图片、JSON 请求和响应，启动探测会验证实际账户的模型访问。Token Plan 面向交互式工具，套餐端点和套餐凭据不用于批量任务。 [百炼服务地址](https://help.aliyun.com/zh/model-studio/base-url) · [Token Plan 使用规则](https://help.aliyun.com/zh/model-studio/token-plan-personal-faq)

## 摆放与材质

- 背景分析按图片内容散列与算法版本缓存，生产模块不解析文件名或相机编号。每张图各自分析料斗内部、侧壁/外部排除区、文字保护区、最多 8 个承托区域、局部尺度参照、相机区间与光照。
- SAM 与相对深度只用于限制候选，不能当作标定深度。区域内按距离变换寻找少量内部落点；再检查真实网格投影、各接触簇、三个俯角和相邻参照深度。跨深度参照差超过 0.085 时拒绝。
- CoACD 生成仅用于碰撞的多个凸部件。封闭部件按均匀密度求体积、质量和质心；碰撞代理保持组件相对位置，渲染网格和 UV 不改写。非封闭面积占比过高、装配关系未知或散件进入待审。
- 每个模型最多六个初始落稳姿态，缓存后搜索水平朝向。长轴与水平夹角不超过 15°；紧凑物体保留低姿态。局部承托坡度改变时单独复验，滚动、穿透、悬空、质心无支撑和未收敛分别记录。
- 真实网格逐像素深度与轮廓保留孔洞。接触斑可以分别落在 2–4 块钢料上；碰撞接收面只覆盖分割片，空隙不填充。前三候选才进行有限承托片的物理复验，底层横躺姿态继续缓存。
- 实际接触簇的投影支撑率达到 ≥90% 才能记为通过；隐藏接触和高度证据不足记为未知。相机中点成图，端点风险保留待审，不再因此丢弃中点合理的检查图。尺度逐锚点重算，跨深度参照不能转为同层证据；缺少参照时使用有明确假设的展示尺度。
- 相对深度拟合到局部承托，再与物体 Z 通道比较，可同时遮挡后方钢料并被前方钢料遮挡。嵌入优先 10%–40%，没有自然遮挡时不制造遮挡。边界在原图尺度细化，前景恢复原图像素。
- 保留模型已有金属/粗糙度/法线贴图和材质分组。混合 PBR 贴图不会因为名称包含“橡胶”而跳过细节；纯塑料/橡胶不增加金属度。新增细节只在渲染材质中产生，源模型文件不变。
- 估计环境反射、柔光源和承托面共同参与渲染；保存 combined、漫反射、高光、Z、金属度、朝上程度以及有影/无影线性 EXR。曝光、白平衡与光照雾化在线性辐射量处理后，只应用一次 Blender OCIO AgX 显示变换；依据见 [Blender 色彩管理](https://docs.blender.org/manual/en/4.5/render/color_management.html)。阴影使用线性 EXR 比值，裁到承托片集合。
- 强融合色度系数限制为 0.20–0.55。所有摄影调整限定在物品 RGB，保持孔洞 alpha。候选先按几何依据排序，渲染首选，明确失败或需要换位时才访问备选，最多三个。最终对物体区域约 2 倍采样，长边为 512–2048，裁剪区域与原相机共用坐标。物体、有物体阴影和无物体基准分别缓存。

这里的接收面是图像推断的有限共面钢料片，**不是实测三维料堆**。高度一致性不足时物理状态仍为未知，不能把仿真稳定等同于真实料堆验证。尺度也是模型尺寸区间与同层参照的估计。只剩尺度/遮挡/隐藏接触未知时默认出待审图；确认穿透、悬空、非横躺和保护区冲突仍拒绝。没有可用基础姿态或模型确有损坏时不会虚构图像。

## 自动复核与输出

AI 查看实际 PNG 的全景、局部和当前模型参照。完整复核同时提供无遮挡物体视图，换位时可附几何诊断。返回原五项评分、材质和关键结构评分、缺陷位置、严重度和证据，以及最多三项动作。程序检查动作是否符合缺陷和生效条件，同轮优先修正同一类原因。

展示合格要求原五项各 ≥3、合计 ≥16，材质与结构各 ≥3，无严重缺陷，并且所有程序检查通过。参数重复、实际像素不变、无有效动作、连续不改善或轮数用完时停止。

在 `native` 模式，视觉服务报告插入痕迹、评分和可见缺陷，程序按规则决定换位、调整照明或摄影参数。只有痕迹判断为 `none` 且既有视觉和程序门槛均通过，才归入 `accepted_display`。`visible` 与 `uncertain` 的成图进入 `review`，汇总单独显示三类痕迹计数。

结构复核区分可见、被遮挡、不可判断。单张图的疑点不会立即隔离模型；只有独立无遮挡渲染对照确认可见的真实损坏才进入模型级待审，该确认按模型缓存。摄影动作新增 `chroma/dust/contrast/haze/sharpness` 的 `auto/less/more` 档位，`more chroma` 表示更强降色度；摄影动作复用渲染通道。

每批目录包括：

| 文件/目录 | 内容 |
|---|---|
| `accepted/` | 展示合格的原尺寸 PNG 与局部预览 |
| `review/` | 最佳已审待审图，包括后续修正执行失败的情况 |
| `jobs/` | 候选、逐轮参数、程序检查、模型意见与阶段结果 |
| `jobs.sqlite` | 持久化队列、检查点与终态 |
| `results.jsonl`、`summary.json` | 全任务记录、按组统计、耗时与本次用量 |
| `config.json`、`inputs.json`、`implementation.json` | 配置、输入散列、代码及依赖版本 |
| `usage-*.json` | 各次执行的真实 API 调用；缓存命中不计调用 |
| `model-review.json` | 当前批次的模型级待审隔离依据 |
| `index.html` | 全部结果的浏览入口 |

结果另存摆放方式、承托片集合、遮挡比例、尺度假设、置信度、摄影参数与逐项拒绝原因。`*-geometry.jpg` 和 `*.occlusion.png` 标注承托片、接触斑和前景。`summary.json` 的 `generated_pngs/image_rate` 与展示合格数分别统计，出图增加不代表物理通过。`review` 与 `failed` 不自动补生成，任务总数固定。保留中间图用于追溯，因此大量批次会占用较多磁盘；源模型、缓存和结果不进入 Git。

GPU 渲染进程使用全局独占锁，默认 OptiX；CPU 作业和 AI 请求各最多 2 路。一个常驻 Blender 复用设备，物理缓存和渲染缓存分开。报告分别记录实际 worker 操作、等待、API 与任务阶段耗时；并行任务耗时之和不能当作批次墙钟时间。

昂贵的资产准备失败也按模型、代码、依赖和服务配置缓存，后续任务进入待审。修复后签名变化会重新准备。复核前保存 `draft-*.json`，记录程序检查与图片散列。首稿复核失败的未审草稿保存在 `jobs/`，后续修正失败时保留最佳已审图。

## 场景融合验证

固定清单覆盖两张背景和三类材质（06 不锈钢罐、13 塑料罐、31 液压缸）。正式 CLI 自动完成选位、物理复验、合成、视觉复核和最多一次修正。选位和修正均由程序执行。实测使用已有 DeepSeek 视觉配置。

| 检查 | 实测 |
|---|---|
| 出图和质量 | 6 张 3840×2160 PNG，6 待审，0 展示合格，0 执行失败。5 张有明显痕迹，1 张无法确认 |
| 证据和像素 | 1 项程序证据通过，5 项未知。6/6 PNG 独立回读通过，保护区、前景和编辑范围外的像素检查通过。44/44 源模型散列相同 |
| 离线复跑 | 新目录重放 6 项，PNG 散列、结果状态和修正记录全部相同，52.047 秒，0 新 API 调用，0 新 GPU 渲染 |
| 本批开销 | 墙钟 145.969 秒，GPU worker 操作 97.469 秒，12 次 API，35,782 已报告 token |
| 本轮累计开销 | 计入前序格式与合成故障试跑，共 39 次 API，138,097 已报告 token |

接触暗部偏弱、表面过平和遮挡层次不足仍影响融合。六个最终位置均为表面放置，本次结果未验证真实批次的嵌入效果。查看[六项完整结果](outputs/scene-native-auto-v4-20261110/index.html)、[像素回读](outputs/scene-native-auto-v4-20261110/png-audit.json)与[离线复跑校验](outputs/scene-native-auto-repeat-20261110/comparison.json)。

## 分阶段验证

使用旧版完整 138 项结果中的已知几何进行固定摆位对照。每次执行使用独立输出目录，`--attempt` 可选 1–3。

```powershell
work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/validate_display_costs.py `
  --stage 12 --attempt 1 `
  --baseline 'outputs/摄影嵌入-验证-v2-20261110/regression' `
  --out 'outputs/下一次摄影验证'
```

阶段 `--stage 36` 使用同一输出目录和通过门槛的尝试编号，`--style-confirmed` 表示整体风格已经确认。执行器核对小样质量及冻结后的素材和渲染签名。阶段 `--stage 138` 还需 36 项验证通过，输入与实现签名一致时复用已完成结果。两次扩批均从各自输出目录续跑。

## 离线诊断与复现验证

```powershell
# 缓存不足时记录 unknown，不调用 AI，也不输出合成图
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py diagnose --config your-diagnostic-config.json

# 明确按相机分组；该编号逻辑仅用于评估清单
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py evaluation-manifests --backgrounds C:/Users/zhaoc/Desktop/背景图 --out work/display-batch-evaluation

# 离线/模拟测试与真实 Blender 物理测试分开执行
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/checks_display_batch.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/checks_display_fusion.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/checks_display_physics.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/placement-web/checks_web.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/audit_display_assets.py --out outputs/asset-audit/audit.json

# 合并真实批次为完整跨背景总览，保留失败任务
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/report_display_batch.py --batches outputs/your-development outputs/your-heldout --out outputs/your-report
```

1000 任务故障恢复测试仅模拟队列，不等于生成 1000 张图片。上一版开发/留出展示合格率分别为 3.3% / 5.6%，见 [历史验证记录](DISPLAY_BATCH_VALIDATION.md)。现有 23 张背景都已查看，本轮 138 项称为跨背景回归；新的独立验收必须另取未参与开发的相机组。不得用通过单张预检代替批量合格率。

固定摆位摄影复验入口为 `replay_display_photography.py --batches <旧批次...> --out <新目录> --rerender-materials`。补齐 EXR 后按版本缓存，固定原 alpha、阴影、位置、尺度与姿态，不重新落稳。`prepare_display_fusion.py --batches <旧批次...> --out <新验证目录>` 生成 48 项恢复清单和 138 项回归清单及配置；两份清单用相同配对种子复用缓存。真实渲染锚点核查使用 `checks_display_renders.py --batch <新批次> --out <新检查目录>`，只读取已有分辨率缓存，不调用模型或新增渲染。

旧批次若在后续复核异常时丢失最佳版本元数据，可运行 `recover_display_evidence.py --batches <已完成批次目录>`。它只按已有检查与评分恢复失败任务的最佳已审图，保留原记录及散列；不调用模型、不渲染、不改变失败状态。独立回读入口为 `audit_display_outputs.py --batches <批次目录> --out <报告 JSON>`。

整轮摄影/恢复/回归汇总入口为 `report_display_fusion.py --root <含 recovery 与 regression 的验证目录> --photo <固定摄影目录> --history <本轮各次尝试目录...>`。它只在两个真实批次均完成后发布最终总览，按每次 `usage-*.json` 汇总真实请求，保留被中断尝试的成本；分别统计视觉达标、程序未知与最终展示状态。跨分辨率检查失败时仍写出全部六类结果及差异图，并以非零码退出，不能按全通过记账。
