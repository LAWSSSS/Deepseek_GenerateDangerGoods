# 多背景批量展示合成

新增流程供人工展示与评审：分析每张背景 → 选择承托区域 → 搜索横躺姿态与朝向 → 渲染 → 同承托面匹配 → AI 复核与受限修正。直接输出原尺寸 PNG，不启动 Photoshop。

`accepted_display` 是新流程的展示状态，不改变已有 `manual_only`、模型人工确认或正式场景验收状态。已有 10 号去独立操作杆、13 号单罐、19 号拆瓶、31 号封闭缸壁修订继续由当前模型目录提供。

## 使用

在仓库根目录运行，先安装新增碰撞分解依赖：

```powershell
work/triposr-env/Scripts/python.exe -m pip install -r outputs/ps-model-controller/display-batch-requirements.txt
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py run --config outputs/ps-model-controller/display-batch.example.json
```

复制示例 JSON 后修改 `backgrounds` 与 `output`。默认 100 个任务、初稿之后最多修正 2 轮；`count` 允许 1–1000，`max_iterations` 允许 0–10。小于 100 的任务数用于诊断。`assets` 留空使用当前全部 44 个条目；可指定 `['3','6','10','13','19-1','31']` 对应的 JSON 字符串数组。每个任务只有一个物品。

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

服务必须支持图片与 JSON function tool 输出，启动时进行真实图像能力探测。图片先编码为有大小上限的 JPEG。每个逻辑请求最多两次尝试，格式错误只修复一次；鉴权、模型端点和视觉能力错误暂停批次。默认不设置整批 token 上限。只有配置输入、输出单价时才计算费用，否则仅报告供应商返回的 token。

## 摆放与材质

- 背景分析按图片内容散列与算法版本缓存，生产模块不解析文件名或相机编号。每张图各自分析料斗内部、侧壁/外部排除区、文字保护区、最多 8 个承托区域、局部尺度参照、相机区间与光照。
- SAM 与相对深度只用于限制候选，不能当作标定深度。区域内按距离变换寻找少量内部落点；再检查真实网格投影、各接触簇、三个俯角和相邻参照深度。跨深度参照差超过 0.085 时拒绝。
- CoACD 生成仅用于碰撞的多个凸部件。封闭部件按均匀密度求体积、质量和质心；碰撞代理保持组件相对位置，渲染网格和 UV 不改写。非封闭面积占比过高、装配关系未知或散件进入待审。
- 每个模型最多六个初始落稳姿态，缓存后搜索水平朝向。长轴与水平夹角不超过 15°；紧凑物体保留低姿态。局部承托坡度改变时单独复验，滚动、穿透、悬空、质心无支撑和未收敛分别记录。
- 实际接触簇的投影支撑率均须 ≥90%，不得把分离足底的空隙当支撑。相机区间两端和中点使用同一物理尺度，最终渲染按接触锚点放回背景。
- 保留模型已有金属/粗糙度/法线贴图和材质分组。混合 PBR 贴图不会因为名称包含“橡胶”而跳过细节；纯塑料/橡胶不增加金属度。新增细节只在渲染材质中产生，源模型文件不变。
- 估计环境反射、柔光源和承托面共同参与渲染；输出漫反射、高光、透明度与有影/无影图。阴影在线性亮度下计算，裁到同一承托面，前景和文字区恢复原像素。
- 调色限制曝光、白平衡、色度、雾化和最终像素噪声；高光由独立 pass 保护，透明度不参与调色。每件最多三个 512 px 粗候选；最终根据整个投影画幅约 2 倍采样，限制为 512–2048。

这里的承托面是图像推断的局部平面，**不是实测三维料堆**。尺度也是模型尺寸区间与同层参照的估计。复杂交叠钢件、不可辨孔洞、隐藏连接关系和细杆仍可能被拒绝，不能通过 AI 分数覆盖程序检查。

## 自动复核与输出

AI 查看实际 PNG 的全景、局部、当前模型参照与已有候选。返回原五项评分、材质和关键结构评分、缺陷位置/严重度/证据，以及最多三项白名单动作。动作只能使用已有位置/姿态/方位角或有限材质、光照、调色档位。

展示合格要求原五项各 ≥3、合计 ≥16，材质与结构各 ≥3，无严重缺陷，并通过程序检查。参数重复、无有效动作、连续不改善或轮数用完时停止；先按硬性检查，再按严重缺陷数量和分数保留最佳版本。明确模型级结构问题在该批次隔离，避免随后逐图修补；AI 的模型级判断仍需人工复核。

每批目录包括：

| 文件/目录 | 内容 |
|---|---|
| `accepted/` | 展示合格的原尺寸 PNG 与局部预览 |
| `review/` | 最佳待审图；执行失败但已有图也保留在这里 |
| `jobs/` | 候选、逐轮参数、程序检查、模型意见与阶段结果 |
| `jobs.sqlite` | 持久化队列、检查点与终态 |
| `results.jsonl`、`summary.json` | 全任务记录、按组统计、耗时与本次用量 |
| `config.json`、`inputs.json`、`implementation.json` | 配置、输入散列、代码及依赖版本 |
| `usage-*.json` | 各次执行的真实 API 调用；缓存命中不计调用 |
| `model-review.json` | 当前批次的模型级待审隔离依据 |
| `index.html` | 全部结果的浏览入口 |

没有位置或姿态满足条件时只保存拒绝依据，不虚构合成图。`review` 与 `failed` 不自动补生成，任务总数固定。保留中间图用于追溯，因此大量批次会占用较多磁盘；源模型、缓存和结果不进入 Git。

GPU 渲染进程使用全局独占锁，默认 OptiX；CPU 作业和 AI 请求各最多 2 路。一个常驻 Blender 复用设备，物理缓存和渲染缓存分开。报告分别记录实际 worker 操作、等待、API 与任务阶段耗时；并行任务耗时之和不能当作批次墙钟时间。

## 离线诊断与复现验证

```powershell
# 缓存不足时记录 unknown，不调用 AI，也不输出合成图
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py diagnose --config your-diagnostic-config.json

# 明确按相机分组；该编号逻辑仅用于评估清单
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/batch_display.py evaluation-manifests --backgrounds C:/Users/zhaoc/Desktop/背景图 --out work/display-batch-evaluation

# 离线/模拟测试与真实 Blender 物理测试分开执行
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/checks_display_batch.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/checks_display_physics.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/placement-web/checks_web.py
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/audit_display_assets.py --out outputs/asset-audit/audit.json

# 合并真实批次为完整跨背景总览，保留失败任务
work/triposr-env/Scripts/python.exe outputs/ps-model-controller/report_display_batch.py --batches outputs/your-development outputs/your-heldout --out outputs/your-report
```

1000 任务故障恢复测试仅模拟队列，不等于生成 1000 张图片。真实评估使用相机 53 的 20 张图开发、相机 36 的 3 张图留出，六类模型共 138 项。完整实际结果与未达标原因见 [WORK_LOG.md](WORK_LOG.md) 最新记录；不得用通过单张预检代替批量合格率。
