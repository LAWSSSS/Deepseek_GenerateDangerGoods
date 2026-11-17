# Photoshop 模型控制器

把文字指令转成经过校验的 Photoshop 操作。可单独配置一个视觉模型，让它对同一局部的修改前后图打分，并返回可直接执行的下一轮操作。像素处理使用已经安装的 [Photoshop Python API MCP](https://github.com/loonghao/photoshop-python-api-mcp-server)。本项目只实现任务协议、受限操作和评分闭环。

## 本机试跑

打开 Photoshop。把本目录放在包含 Scrap-Ambient-Bright.psd 的目录下，或者修改 task.example.json 的 source 指向实际 PSD。以下命令使用本机此前为 Photoshop MCP 创建的 Python 环境。在本项目目录运行。

```powershell
$psPython = Join-Path $env:LOCALAPPDATA 'OpenAI/PhotoshopMCP/python/Scripts/python.exe'
& $psPython .\run.py --task .\task.example.json --plan .\plan.example.json --out .\demo-offline
```

--plan 读取现成的结构化计划，适合在没有模型凭据时验证 Photoshop 连接和目标选区。每次运行指定一个新的输出目录。已有输出目录会报错，以便重复运行时清楚地区分结果。

## 接入模型

1. 复制 config.example.json 为 config.json。为 planner 填写文字模型的完整 chat/completions 地址和模型名。需要视觉复核时，为 vision 填写视觉模型的独立地址和模型名。只做一次修图时，将 vision 设为 null。
2. 供应商要求认证时，在对应配置加入 "key_env": "PS_PLANNER_KEY" 或 "key_env": "PS_VISION_KEY"，并通过本机环境变量提供对应密钥。无需在 JSON 中填写密钥本身。
3. 复制 task.example.json 为 task.json，指定源图、指令、目标区域和允许的操作。随后运行以下命令。

```powershell
& $psPython .\run.py --task .\task.json --config .\config.json --out .\run-01
```

接口使用兼容 chat/completions 的文本和视觉请求格式。配置模板使用本地 [Ollama 兼容接口](https://docs.ollama.com/api/openai-compatibility)。云端可以填支持同一协议的第三方 HTTPS 地址，例如支持图像输入的 [阿里云百炼视觉接口](https://www.alibabacloud.com/help/en/model-studio/vision)。视觉端需接受 JPEG data URL。供应商不支持 JSON mode 时，可把该端的 json_mode 设为 false，客户端仍严格校验返回的 JSON。输出长度参数可以通过 token_field 在 max_tokens 和 max_completion_tokens 之间切换。支持 reasoning_effort 的供应商可以在对应配置中设为 "none"，以减少推理开销。

若在其他 Windows 电脑运行，先按上游说明安装 Photoshop Python API MCP，在该 Python 环境中启动 run.py，并用 --mcp 指向 photoshop-mcp-server.exe。当前方案处理 8 位 RGB 的 PSD、PNG、JPEG 或 TIFF。

## 任务与操作

- source 是相对任务 JSON 的路径，也可用绝对路径。程序从磁盘复制源文件作为本次工作的输入。
- targets 把目标编号映射到 polygon。每点是 [x/图宽, y/图高]，左上角是原点。羽化范围由 feather_px 设定。模型只接收编号和简短名称。新图片需要提供对应选区，示例已标好储罐。多个目标用不同编号。允许多个目标时，视觉预览会覆盖它们和周边参照。
- group 和 hide_groups 可用来指定现有 PSD 的一个物品图层组，并隐藏其他测试物品。普通图片省略这两个字段。
- allowed_ops 限定可执行的操作。limits 可收紧数值范围，例如将 brightness 限为 [0, 25]，以便强制这一轮只允许提亮。
- preview_rect 可固定视觉比较范围，使用 [left, top, right, bottom] 四个归一化坐标。省略时按目标范围自动取局部，并留周边环境参照。

| 操作 | value 的含义 | 全局允许范围 |
| --- | --- | --- |
| brightness | 8 位曲线的中间亮度增减 | -48 到 48 |
| warmth | 红蓝通道的相反方向偏移，正值偏暖 | -24 到 24 |
| saturation | Photoshop 饱和度增减，整数 | -40 到 40 |
| blur_px | 高斯模糊半径，原图像素 | 0.2 到 5 |

模型操作示例。

```json
{"status":"ready","ops":[{"target":"tank","op":"brightness","value":14}],"note":"提亮罐身"}
```

视觉复核示例。before 和 after 是同一评分口径下的 0–100 分。next_ops 针对通过后的版本，或者未通过时仍保留的版本。

```json
{"before":72,"after":86,"regression":false,"next_ops":[],"note":"罐身更接近相邻钢板"}
```

输入和模型响应的 JSON Schema 在 schemas.json。程序校验目标、操作、数值、字段和每轮操作数量，再将参数交给固定的 Photoshop 模板。超出能力的指令由规划端返回 unsupported。目标名称和选区的对应关系由任务提供。评分是视觉模型的相对判断，不能替代对素材用途和编辑结果的人工确认。

## 迭代、成本和输出

文字模型每个任务调用一次。启用视觉后，每次试改最多调用一次视觉模型，默认最多两次试改，上限四次。视觉调用只带任务摘要、当前操作和两个同范围的 JPEG，默认最长边 640 像素，每张最多 384 KiB。每次请求都有输出 token 上限。实际输入 token、计费和隐藏推理开销由模型供应商决定，result.json 记录供应商返回的用量。

达到 target_score、视觉端不给后续操作或达到轮数上限时结束。试改只有在分数至少增加 min_gain 且未被标记为有意外改动时才接受。未通过、视觉端报错或后续动作不合法时保留已经接受的最佳版本。视觉为云端时，选区及其周边 JPEG 会发送到你配置的视觉端。

- final.psd 保留原有图层及新增的蒙版和调整图层。final.png 是交付图。
- result.json 记录状态、实际执行操作、每轮评分、是否接受、模型用量和输出路径。CLI 也输出同样的 JSON，错误退出码为 2。
- completed_without_review 表示只执行了结构化计划。target_score_reached 表示视觉端判定达标。best_retained 表示试改结束并输出当前最佳版本。review_failed_kept_best 或 invalid_next_ops_kept_best 表示保留可用版本，同时记录中止原因。
- mcp.log 保存本地 Photoshop MCP 的诊断信息。
## 让模型决定异物摆放

plan_placement.py 把背景局部预览和已有透明物品预览发给配置中的 planner。此处的 planner 必须支持图像输入。预览可缩小，背景预览的构图必须与原图裁区一致。render_placement.py 读取模型返回的坐标、尺寸、角度、曝光和遮挡多边形，用 Photoshop 把现有透明 PNG 放到原图上。它保存独立的物品、阴影及遮挡图层。两步的 JSON 可用于复现。

```powershell
& $psPython .\plan_placement.py --config .\config.json --scene .\scene-preview.jpg --asset .\asset-preview.jpg --out .\placement.json
& $psPython .\render_placement.py --background .\background.jpg --asset .\cutout.png --crop 1480 720 1260 900 --plan .\placement.json --out .\placed
```

crop 依次是背景预览在原图上的左上角像素坐标和裁区宽高。遮挡点以裁区归一化坐标表示。复杂钢筋的边缘仅靠视觉模型回传坐标可能不准。可以将已摆放版本的同裁区预览通过 --trial 传回规划端，必要时通过 --instruction 约束遮挡。后续光照和质感调优可以将 placed/final.psd 作为任务源文件，并用目标可见轮廓定义选区。

## v3 废钢监控两图入口

需要 Windows、Python 3.12 和已连接的 Photoshop Python API MCP。启动 Photoshop。首次安装可运行安装脚本建立 CPU 环境并校验权重哈希。官方源无法访问时加 `-UseMirrors`。已安装的环境和已下载权重可以直接传入命令。规划与复核可用不同的服务，两端都需接受 JPEG 和 JSON 输出。DeepSeek 示例读取环境变量 `PS_DEEPSEEK_KEY`。

```powershell
.\setup_v2.ps1 -RuntimeDirectory .\runtime
$batchPython = '.\runtime\venv\Scripts\python.exe'
& $batchPython .\batch_v2.py --object 'asset-cd0a8343411b' --background '.\frames\53.jpg' --config .\batch-config.example.json --out .\result --weights .\runtime\weights --cache .\cache
& $batchPython .\batch_v2.py --manifest .\jobs.jsonl --config .\batch-config.example.json --out .\batch-result --weights .\runtime\weights --cache .\cache
```

物品参数可用素材 ID 或图像路径。`asset-catalog.json` 记录了 39 张素材的哈希、照片型号、外廓毫米区间和厂家依据。已做旧的对应图可按独立哈希关联原始记录。迁移目录时用 `--asset-root` 指定原图根目录，也可用 `--catalog` 加载另一套库存。同一类别的图各自保留尺寸。入库无有效尺寸或哈希未知时进入待审。明确提供 `object_size` 可用于新增物品。抠图、分层做旧和候选识别点以内容哈希缓存，首次使用时生成。

### 批次输入

JSONL 每行一个任务，路径相对清单所在目录解析。尺寸按实际拍摄状态填写，外廓长轴和短轴单位均为毫米，单值重复写成区间。已标定参照提供实际跨度、依据和原背景上两个端点。坐标左上为原点，横纵各自归一化至 0–1000。`objects` 多视图和 `mode:"preview"` 继续走旧版入口。多视图应属于同一实物、同一伸缩状态。

```json
{"job_id":"53-cylinder-001","object":"asset-cd0a8343411b","background":"frames/53.jpg"}
{"job_id":"53-cylinder-002","object":"assets/cylinder.png","background":"frames/53.jpg","object_size":{"major_mm":[580,580],"minor_mm":[92,92],"basis":"measured","source":"CYL-012 实测"},"scene_references":[{"id":"G1","label":"同层槽钢边","span_mm":[1000,1000],"points":[[370,580],[590,585]],"basis":"measured","source":"53-G1 现场标尺"}]}
```

示例只说明格式。完整输入见 `batch-job.schema.json`。纯估算场景要求同一候选区有两件独立钢件、深度和投影方向可比、比例区间交集非空，且物品投影区间上限与下限之比不超过 1.8。有可核验参照时至少需要一件局部有效参照。新入口的物品大小由实物毫米范围和局部尺度决定。旧草稿的 `width_range` 以全图宽度的比例填写。

### 视觉接口与质量检查

MobileSAM 和 Depth Anything V2 在本地分析分区、钢件端点和相对深度。规划端只返回 `zones`、`refs`、`protect`。每条参照为 `[ID,类别,毫米端点一,毫米端点二]`，程序按从小到大排序，并检查独立性、局部深度和尺度交集。只有校验通过才搜索落点。复核端从实际渲染的至多三组同裁区对照中返回 `pick`、`scores`、`defects` 和可选 `fix`。五项按承托、遮挡、光色、边缘、监控质感排序，每项至少 3/4、总分至少 16/20。轻微亮度、色偏、饱和度、清晰度问题可修一次。接口见 `batch-vision.schema.json`。

本地用承托附近原像素计算亮度分位、对比、Lab 色偏、色度、清晰度和噪声。PSD 底层是完整原图，中间是物品、接触阴影和剪贴调色组，顶层只复制同一底图实际压住物品的钢件像素，裁去无关部分。用 Photoshop 实际输出的两层透明度核对 40%–85% 遮挡、原图边缘对齐、承托与特征露出。顶部 9%、底部 18% 及识别出的其它叠字区域均保护。交付前逐像素检查这些区域、完全遮挡处和编辑范围之外的背景。

同一背景内容和配置只做一次场景分析。每张通过渲染预检的任务调用一次复核，可执行修正最多追加一次。单次上传一张不超过 384 KiB 的 JPEG，输出上限 192 token。每次调用预留 4,096 token，按服务回报结算。批次默认硬上限 200 万 token，使用 `--budget` 可降低预算。

### 批次输出与续跑

`results.jsonl` 和每项的 `result.json` 包含尺寸来源、参照与投影区间、实际位置、摄影测量、修正、评分、用量及淘汰原因。结构见 `batch-result.schema.json`。用 `index.html` 查看结果、筛选状态和分数。

- `status:"passed", delivery:"formal"` 交付原尺寸 `final.png` 和分层 `final.psd`。
- `status:"review", delivery:"estimated_preview"` 表示通过几何和视觉门槛、尺度为估算。输出 `estimated-preview.png` 和 `estimated-preview.psd`。
- 其他待审或失败项记录原因。用 `--keep-review` 可留存通过本地几何但视觉待审的 `review-only.png` 和三层 `review-only.psd`，用于检查和手工调整。

Photoshop 串行渲染，批次按背景内容分组。任务、素材、配置或版本改变会生成新记录。重跑同一输出目录会复用已交付项。`--retry` 重做其余待审或失败项。运行 `pilot_review.py --batch .\batch-result` 生成手工验收表，`accept` 和 `severe` 填 `yes` 或 `no` 后可重新统计。

离线回归检查用 `checks_v2.py`。运行 `checks_photoshop.py --cache .\cache` 可验证已知尺寸、三层顺序、完全遮挡处的像素、保护区和估算 PSD。

## 固定场景人工尺寸标注

`scale_trial.py` 按一份固定背景配置遍历物品库。带尺寸依据的素材使用统一的试摆像素比例。缺少尺寸的素材读取配置中逐张指定的完整长轴像素。生成原尺寸 PNG、全图预览和统一倍率局部，不调用模型。批次标记为人工尺寸评审，可在 `index.html` 标注合适、偏小、偏大或看不清，并导出 JSONL。`suggested_major_px` 指完整物品轮廓的长轴。

```powershell
& $batchPython .\scale_trial.py --scene .\场景配置.json --out .\尺寸评审 --weights .\runtime\weights --cache .\cache
```

已生成的高清素材可配合 `asset-catalog-hd.json` 使用。把 `--catalog` 指向该文件，`--object` 填对应原貌或生锈 PNG。目录中标记为 `prepared_generated` 的透明图直接进入合成，保留输入边缘和表面纹理。生成图关联原素材的尺寸依据，进入物理尺度流程时按估算处理。

## 固定监控场景的空间姿态复测

`pose_retest.py` 从上一轮 `results.jsonl` 和用户填写的 `复测标注.csv` 读取投影尺寸。固定相机下先筛承托和相对深度。Photoshop 从独立物品图和背景图制作合成，接触阴影及真实前景遮挡使用独立图层。

需要补拍或生成独立物品视角时，按顺序运行 `prepare`、`select`、`import` 和 `render`。四个阶段共用 `--before`、`--scene`、`--out`、`--weights` 和 `--cache`。`select` 的视觉配置默认读取 `batch-config.example.json` 和进程环境中的 `PS_DEEPSEEK_KEY`。`生成请求.jsonl` 只包含物品参考图和姿态文字。用 `import --numbers N --view PNG` 导入四周有透明边的独立物品图。

默认复测 03、05、07、10、27、30。对照页记录候选证据、样片和待审原因。验收通过的样片在 `试产验收.csv` 中填写 `size_ok=yes`、`natural_ok=yes`、`severe=no`，随后可以用 `--all` 批跑。阶段结果按素材、标注和场景内容识别，可续跑。

四件样片在 `prepare` 和 `render` 命令中加 `--open-from <已验收尺寸的空间姿态复测目录>`，并用独立的 `--out`。该入口只处理 03、10、27、30，读取上一轮实测且标为“合适”的长轴，直接使用现有独立物品图和固定底图，在本地运行。先找各承托部位足迹至少 90% 落在同一平面且留有净空的位置。空位允许 0% 遮挡，空位预检失败后才检查堆内位置。

程序按独立物品图可见的轮廓对齐角度和大小，从承托面附近取样匹配颜色与纹理，按各部件的受力边界制作短接触阴影。Photoshop 输出待审 PNG 和分层 PSD，并逐像素检查监控叠字与合成范围之外的底图。结果记录 `composite_method`、`placement_mode`、实测长轴、源图旋转、足迹、净空和遮挡。待审图可用 `render --refresh-held --numbers N` 重新渲染。

四件程序合成会根据周围钢板限制物品的色度、反光和标识对比度，保留着色前的独立物品图层。加 `--quiet-from <上一轮程序合成目录>` 可在相同坐标、轮廓和尺寸下对照调色结果，输出须使用另一个目录。

需要重新选位时，用 `--reposition-from <上一轮程序合成目录>` 对照。先筛承托与净空，再优先选择靠近钢件边缘、周围有纹理且反光较弱的空位。直立件会用实拍轮廓补查窄空位。反光偏强的空位作为备选，废钢堆内的候选须有原图钢件的遮挡依据。结果记录新旧位置的纹理、反光、边缘接近度和程序排序分数。

固定场景的 `auto` 阶段连续完成候选准备、选位和 Photoshop 出图。程序用承托面的局部相对深度消除斜坡影响，检查物品投影周边相对两侧是否有更近的钢件。通过承托、净空和叠字检查后，优先选有双侧证据的凹位，淘汰有明确证据的凸位。深度证据不足会单独标记。通用批次的遮挡候选同样使用这项深度检查。

启用 DeepSeek 后，只有两个同级候选差距很小时，固定场景才发送一张低细节拼图，让模型调用一次 `choose_slot`。函数只接受 A、B 或跳过，程序校验后直接执行，输出上限 64 token。相同候选复用缓存。固定场景的 `token-usage.csv` 和批次的 `index.csv` 逐图列出 API 用量，`results.jsonl` 同时记录输入、输出和缓存命中用量。相同版本的重试累计到该图，无回执的请求记为用量未知。共享场景分析只记入实际发出请求的图。

## 四件样片：三维重建试验

这条试验路线独立于上面的二维批次。它使用 TripoSR 的固定版本在 CPU 上重建网格，Blender Cycles 渲染，并用 DeepSeek 核对原素材与网格。单图背面和孔洞深度属于推断；轮廓、孔洞、细部、每个独立部件的落稳、源图可见面积、尺度、背景保护和视觉验收中任一项证据不足时，结果为待审。正式合成入口仍只接入固定场景的 03、10、27、30；任意两图及 JSONL 不会自动使用三维路线。39 件素材的模型库和场景选角独立入口见项目根目录 README 的“三维模型库与场景选角”。

先按已有 `setup_v2.ps1` 安装 v2 运行时。然后从工作区根目录运行 `work/v2-env/Scripts/python.exe tools/ps-model-controller/setup_mesh.py`，安装固定版本的 TripoSR、Blender 和相关依赖。模型权重下载后会核对 SHA-256。该脚本不读取或保存 API 密钥；DeepSeek 密钥仍通过进程环境变量 `PS_DEEPSEEK_KEY` 配置。

下面的命令从缓存重建四件模型，按统一策略修补原图可见孔洞，并对通过结构检查的模型保拓扑减面。模型、原图、相机和脚本内容改变会生成新的缓存签名。每个重建进程限时 900 秒；Windows 作业对象将进程树的提交内存限制在 10 GiB，并额外监测常驻内存。模型处理和 Photoshop 串行执行。

```powershell
$meshPython = (Resolve-Path 'work/triposr-env/Scripts/python.exe').Path
$meshCode = (Resolve-Path 'tools/ps-model-controller').Path
$blender = (Resolve-Path 'work/blender45/blender-4.5.3-windows-x64/blender.exe').Path
$old = 'outputs/固定场景-39件高清复测'
$scene = 'outputs/固定场景-39件尺寸标注/场景配置.json'
$approved = 'outputs/固定场景-空间姿态试产'
$baseline = 'outputs/固定场景-凹位优先复测'

& $meshPython "$meshCode/mesh_build.py" --before $old --scene $scene --out 'outputs/3d-build' --weights 'work/v2-models' --cache 'work/v2-cache' --open-from $approved --reposition-from $baseline --vendor 'work/vendor/TripoSR' --python $meshPython --blender $blender
& $meshPython "$meshCode/mesh_revalidate.py" --before $old --scene $scene --out 'outputs/3d-camera' --previous-build 'outputs/3d-build' --cache 'work/v2-cache' --open-from $approved --reposition-from $baseline --blender $blender
& $meshPython "$meshCode/mesh_repair_batch.py" --build 'outputs/3d-camera' --out 'outputs/3d-repair' --cache 'work/v2-cache' --python $meshPython --blender $blender
& $meshPython "$meshCode/mesh_optimize_batch.py" --build 'outputs/3d-repair' --out 'outputs/3d-optimize' --cache 'work/v2-cache' --python $meshPython --blender $blender
& $meshPython "$meshCode/mesh_pipeline.py" --build 'outputs/3d-optimize' --scene $scene --approved-from $approved --out 'outputs/3d-pilot' --weights 'work/v2-models' --cache 'work/v2-cache' --blender $blender --budget 2000000
```

新背景分析和每个独立物品的结构检查按哈希缓存。场景俯视角驱动 Blender 的实际相机；候选使用同一个网格进行刚体模拟和渲染，支撑图允许上层部件由其他已落稳的部件支撑。安全空位允许零遮挡。当前堆内钢件体积与物品的碰撞深度仍不能可靠校验，堆内候选会待审，不会被当作已验证的遮挡输出。输出的 `results.jsonl`（结构见 `mesh-result.schema.json`）、`index.html` 和 `token-usage.csv` 记录每图调用量、模型缓存、重建及模拟失败原因；未产生合格候选时不会创建冒充成品的 PNG 或 PSD。

离线验证运行 `checks_mesh.py --blender <blender.exe>`；Photoshop 图层、投影尺寸及背景保护的真实应用测试运行 `checks_mesh_photoshop.py --cache <单独的测试缓存目录>`。两者使用合成测试件，不调用 DeepSeek。

### 在新设备继续四件样片

迁移包可用项目本地 Python 3.12 初始化：`.\setup_handoff.ps1 -PythonExecutable <python.exe>`。已有标定网格时，可直接从 `mesh_repair_batch.py` 开始；孔洞雕刻现在按投影体素的覆盖范围清理细孔边缘，仍以 Blender 的真实回渲检查孔洞、轮廓和细部。

`mesh_repair_batch.py` 和 `mesh_pipeline.py` 接受 `--render-device OPTIX` 或 `CUDA`。默认 CPU；显式选择的后端不可用时记录失败。这个选项只加速 Cycles 渲染，刚体落稳和 TripoSR 重建仍使用 CPU。

`mesh_pipeline.py --geometry-only-from <同一底图的旧三维结果目录> --budget 0` 复用旧场景俯角，仅检查四件网格的落稳和可见面。它不调用 DeepSeek、不选场景位置、不生成合成 PNG 或 PSD，结果一律待审。正式复测使用新输出目录，去掉 `--geometry-only-from`，通过 `--mcp` 指定 Photoshop MCP；补齐 DeepSeek 配置后可用 `--retry` 重跑待审项。密钥只从本地进程环境读取。

可见面检查在源图原分辨率登记有依据的网格面。默认方向全部未通过时，程序在已落稳的姿态上按 30° 间隔估算水平朝向，最多重新渲染一个候选。候选仍需通过实际渲染的轮廓一致性和 80% 可见面检查。
