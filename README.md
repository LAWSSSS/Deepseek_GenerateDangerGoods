# 废钢监控异物合成

从完整迁移包恢复的 Windows 项目。代码在 `outputs/ps-model-controller`；原图、模型、历史结果和虚拟环境保存在本机工作区，Git 只跟踪代码、配置示例和运行说明。查看 [控制器说明](outputs/ps-model-controller/README.md)。

39 件素材可建立可续跑的三维模型库，并按场景选择单件视角。原有正式合成试产仍使用 **03 轴承座、10 千斤顶、27 减震器、30 液压缸**；模型库中的待审模型不代表已通过正式合成验收。三维结构、承托、净空、源图可见面和视觉验收的门槛独立生效。

## 本机人工放置网页（待审版本）

从项目根目录安装本地网页依赖并启动：

```powershell
Push-Location outputs/ps-model-controller/placement-web
npm ci
Pop-Location
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/placement-web/server.py
```

在本机浏览器打开 `http://127.0.0.1:8765`。选择监控底图或上传图片，再选择一件模型。物体可拖动；右上圆点旋转视角，右下方块缩放。选择“识别可能的遮挡”后，悬停查看分割区域，勾选位于前景的区域，并可用圆点选择局部调色的承托面。生成预览后可以调用 Photoshop 导出 PNG 和分层 PSD。页面上的“自动放置”入口启动原有位置优先流程，默认模型预算为 0。

网页只监听本机。对话需在启动网页的进程中配置 `PS_DEEPSEEK_KEY`，并会将缩小的背景及区域标识发送给 DeepSeek；不要把密钥粘贴到聊天或提交到 Git。单次运行采用 30000 token 预算及调用前预留；网络异常且用量未知时保留预留额度。没有密钥时仍可手动选择遮挡区域。

这是一版**人工检查工具**：交互模型和 Blender 基础渲染使用均衡中性光，网页按原模型的 Z 轴向上转换坐标。程序尝试识别承托区域并估算亮度、色偏和屏幕空间明暗变化；没有明确承托区域时使用周边像素，仅作为待审近似。可手选承托面覆盖估计。按最终输出尺寸进行受限的锐度匹配，保留透明边缘和孔洞。Photoshop 保留隐藏的中性原始物体图层及环境匹配后的物体图层。单张照片的亮度梯度不能证明三维光源方向。网页与 Blender 的轮廓会交叉核对，未通过时停止；但当前网页尚未接入已验证的落稳姿态、接触点、现场尺度、保护区以及接触阴影的完整验收。前后关系需要人工或对话确认；相对深度不是可证明的遮挡顺序。输出均标为待审，不能当作正式验收结果。39 个原模型都可浏览，其中 15 件源视图结构待审。

03 轴承座另有“六视图背面修复候选”供人工对照。该候选保留原图约束，依据未标定的参考图推测背面对称；源视图轮廓和孔洞通过检查，顶部过渡接缝、背面细节与物理姿态仍待复核。候选不替换原模型，不能用于自动放置。

03 ASAHI P210 轴承座另有依据用户六视图建立的分部件重建 v1（`bearing03-parametric-v1`，目录 `outputs/03轴承座-六视图重建-v3`）。启动网页后访问 `http://127.0.0.1:8765/review/03`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、关键局部、隔离开口、同光照材质、变光和金属反射对照。候选保留轴承内圈/外圈、两道密封、两个紧定螺钉、顶部油嘴、ASAHI/P210 铸字、两个安装长槽、抬高的槽口凸台、底部加工垫和底部拱。参考图未标定且拼图存在横向压缩/纵向拉伸；UCP210/P210/UC210 公开尺寸比例只作合理性参照，不是对实物的测量。原模型源图检查仍有填孔，新候选的单张做锈图诊断（IoU 0.6673，1 个孔被填）未通过，均只作诊断；背面、内圈轴向偏置、紧定螺钉角度和隐藏面仍待人工复核。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成；网页核验只覆盖加载、切换、灰模/材质与拖动旋转。

01 长阀杆阀门另有依据用户提供六视图建立的分部件重建候选，以及在此基础上改善金属度、粗糙度的 v7 材质版。启动网页后访问 `http://127.0.0.1:8765/review/01`，可旋转切换原模型、上一版材质、新版及灰模，并查看同一中性光下的材质对照。新模型不会替换原模型或进入自动放置。参考图没有标定，侧背面材质和不可见结构为推断；原图轮廓诊断未通过，模型真实性、现场尺度、落稳及合成均待人工检查。

02 DN200 球阀另有依据同一六视图建立的分部件重建 v1。启动网页后访问 `http://127.0.0.1:8765/review/02`，可旋转切换原模型、新版和灰模，并查看六视图新旧对照、局部检查、隔离开口检查及材质对照。候选保留三辐手轮的三个开口、12 颗剖分法兰螺栓、两端各 8 个法兰螺栓孔、球芯、底部六角盲孔和双立板支脚；隔离渲染检测到 3/9/2 个开口。原模型源图诊断仍有填孔，新候选的单张做锈图诊断也未通过，均只作诊断；参考图未标定，剖分法兰左右方向、支脚形式、球芯深孔光照、OPEN/SHUT 凸字和背面细节仍待人工复核。候选为 `manual_only`，不替换原模型，不能进入自动放置或正式合成。

从本地六视图重建该候选（输出目录须为新目录；相同签名的完整结果可复用）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item.py --number 1 --reference "$env:USERPROFILE/Desktop/六视图/工业阀门六视图目录图.png" --out 'outputs/01阀门-六视图重建-v7'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_01.py
```

输出含可编辑 Blender 文件、GLB、2048 px 材质图集、六方向与同光照材质对照、24 张转台视图及 JSON 检查记录。v7 默认读取保留的 v6 模型制作同光照对照，需保留其输出目录。四个手轮开口和八个底部安装孔通过独立部件透明度检查；同部件使用 BVH 检查非相邻面交叠。检查不能证明未观察部分或部件之间的装配关系。01、02 各有独立入口，其他物品须逐件建立对应结构与参考。

39 件的中性光六方向检查图与逐件记录位于 `outputs/三维模型-39件六视角审计`。网页提供 08 分离过滤器与 19 气瓶组的漂浮碎片修复候选、29 减震器的源图孔洞修复候选，以及 15、16、36 号喷罐的圆柱截面修复候选。源图检查通过后才显示为可选，但隐藏结构、部件关系和落稳仍待审，原模型保留。其他模型的缺陷不因六方向渲染而自动视为修复。

网页预览为金属材质提供中性反射环境。插入时尝试从多个方向的细钢件明暗提取低置信度的屏幕光向；线索不足或不一致时使用中性光。该估计依赖圆柱形、表面颜色近似均匀等假设，不能视为实测三维光源。细钢件额外使用原图尺度的线段候选与受限分割，前后关系及边界仍需确认；选择失败的候选不会被当成已验证遮挡。

离线回归（不调用模型）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/placement-web/checks_web.py
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_mesh.py
```

## 三维模型库与场景选角

在项目根目录运行（第一次重建会耗时，重复命令从逐件记录续跑）：

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_library.py --retry
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_select_view.py --view-only --asset 3 --background '现场底图.jpg' --out 'outputs/选角结果' --target-px 224.6
```

模型库按来源哈希复用已有模型，逐件记录原视角轮廓、孔洞和细结构；做锈素材中的孔洞仅在原貌素材的透明区域提供对应依据时尝试一次受限修复，失败时保留原始网格和拒绝原因。单张图片中不可见的表面及孔深属于推断，关键结构仍需复核。

`--view-only` 保留原先只输出视角的诊断流程；它读取场景的深度、承托区、保护区和俯角，比较落稳且源图可见面不少于 80% 的姿态，并复核俯角区间端点。`selected-view.png` 是独立视角而非已验收合成；`result.json` 记录依据及待审项。未提供 `--target-px` 时使用画面比例产生候选尺寸，该值不代表现场实测。缺少所需证据或没有合格候选时停止并记录原因。

场景分析复用按背景和配置签名的缓存；新场景没有缓存时需要配置视觉模型和本机密钥，并传入 `--budget`（最高 30000）。默认预算为 0，不会调用外部模型。

## 先选位置、再选视角和调色

`mesh_select_view.py` 默认使用位置优先流程，也可直接运行 `mesh_location_first.py`。程序先根据场景承托区、净空、保护区及局部参照排序接触位置，再在每个固定位置检查真实三维视角。候选还需通过落稳、源图可见面、各接触斑、实际轮廓和俯角端点检查；不合格时自动尝试下一个位置。确定位置和视角后才从同一承托面取样调色，排除前景、阴影和保护区，并验证 Photoshop 实际导出。

```powershell
$scene = Get-Content 'outputs/固定场景-39件尺寸标注/场景配置.json' -Raw | ConvertFrom-Json
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_location_first.py `
  --asset 03 --background $scene.background --target-px 224.6 `
  --settled-view 'work/03-smooth-poses/pose-45' `
  --previous 'outputs/轴承座-圆润后场景检查-v4' --out 'outputs/位置优先-新运行' --budget 0
```

`--settled-view` 可选；仅在网格完全一致且单部件落稳通过时复用物理结果，程序会重新投影接触点。省略时会从当前模型重新模拟。`--previous` 可选，只用于输出对照图，并检查其物品与背景。每次使用新输出目录。结果包括分层 PSD、全图 PNG、对照图、位置与视角诊断；估算尺度和缺少的结构、视觉复核仍会标为待审。

## 六视图参考与圆润处理

六视图拼图按物品外观关联，关联清单位于 `outputs/ps-model-controller/six-view-map.json`。参考图没有相机标定，部分是相似类别或包含多个物品，不能直接作为精确几何。程序记录这些不确定性；目前使用它们进行视觉对照，并未进行多视图重建或补齐不可见结构。

```powershell
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_smooth_library.py `
  --six-views '<六视图文件夹>' --out 'outputs/三维模型库-圆润复验'
& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_smooth_report.py
```

平滑限制顶点位移，并保护源视图边缘；输出前检查面翻转、连通性、纹理、体积及源视图中的轮廓和孔洞。未通过保护门槛时保留原模型。局部起伏指标仅用于量化变化，不表示外观或隐藏结构已经验收。使用圆润模型进行场景选角时，给 `mesh_select_view.py` 增加 `--library 'outputs/三维模型库-圆润复验/results.jsonl'`；几何发生变化后应重新验证落稳，不复用旧的刚体姿态。

轴承座示例可通过 `mesh_sixview_review.py` 将同一组近似方向上的平滑前后渲染与参考拼图并列。图中 IoU 是新旧模型之间的轮廓一致性，不是对参考图的精确匹配。使用 `mesh_pipeline.py --inspection-reference <上一版/result.json>` 可以保留旧图作为对照，同时允许重新搜索实际接触点；`--inspection-baseline` 则继续锁定原接触点。

在工作区根目录打开 PowerShell，使用已经安装的环境和减面结果运行：

```powershell
. .\enter_handoff.ps1
$py = 'work/triposr-env/Scripts/python.exe'
$code = 'outputs/ps-model-controller'
$mcp = 'work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe'
$blender = 'work/blender45/blender-4.5.3-windows-x64/blender.exe'

$common = @(
  '--build', 'outputs/固定场景-投影孔洞减面复验',
  '--scene', 'outputs/固定场景-39件尺寸标注/场景配置.json',
  '--approved-from', 'outputs/固定场景-空间姿态试产',
  '--weights', 'work/v2-models',
  '--cache', 'work/v2-cache',
  '--blender', $blender,
  '--mcp', $mcp,
  '--render-device', 'OPTIX'
)
& $py "$code/mesh_pipeline.py" @common --out 'outputs/固定场景-朝向与承托复验' `
  --geometry-only-from 'outputs/固定场景-三维支撑关系终测' --budget 0
```

正式复测需要在运行它的本机进程中提供 `PS_DEEPSEEK_KEY`，启动 Photoshop，然后执行：

```powershell
& $py "$code/mesh_pipeline.py" @common --out 'outputs/固定场景-三维续测' --retry
```

复测报告在各输出目录的 `index.html` 和 `token-usage.csv`。本地诊断不会调用 DeepSeek 或交付合成图片。正式流程只对通过必要检查的候选使用 Photoshop，并将真实合成预览交给 DeepSeek 复核。

## 三维 v2 验证

可见面按逐像素深度和源视图表面回投计算；256、512 两种分辨率相差超过 2 个百分点时结果为未知。网格部件使用均匀密度体积质心与相对质量；非封闭网格、装配关系变化和未通过支撑的姿态保留待审。场景候选先筛选可用尺度参照，实际移动后再次核验尺度。正式流程要求分阶段结构复核，历史 v1 模型结论不会自动视作 v2 通过。

`--geometry-only-from` 仍是离线姿态诊断入口：需要同一背景的历史场景俯角，设 `--budget 0`，不会运行结构模型、场景选位或 Photoshop。输出中的 `checks` 记录已完成检查与未知项。正式复测仍需要本机进程环境变量 `PS_DEEPSEEK_KEY`，并以 `--budget 30000` 限制本轮模型用量。

正式批处理中的自动结构缺陷修复、独立阶段诊断开关和物理与渲染分离缓存仍待接入。

## 03 轴承座单件检查图

轴承座检查流程会比较 48k、96k 面网格，对源图孔沿和外轮廓进行保护，生成 2048 像素纹理，并从素材提取受限的表面细节和微小起伏。不可见表面的纹理仍是投射推断。正式尺寸参考仍按来源标记为估算或测量。人工检查图可在参照不足时标记待验，正式流程仍要求完整尺度与模型复核。

在前面的 PowerShell 设置之后，可从同一批次重新生成网格和姿态，再输出到一个新目录：

```powershell
$prior = Get-Content 'outputs/固定场景-v2最终几何复测/results.jsonl' -First 1 | ConvertFrom-Json
$camera = 'work/03-source-camera.json'
$prior.reconstruction.source_camera | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $camera
$alpha = Join-Path (Split-Path $prior.source_view) 'source-alpha.png'
$detail = 'work/03-detail'
$poses = 'work/03-poses'
& $py "$code/mesh_detail.py" --mesh $prior.pre_optimization_mesh --alpha $alpha --camera $camera --asset $prior.rust --out $detail --blender $blender --render-device OPTIX
if ($LASTEXITCODE -ne 0) { throw '网格细节处理失败' }
& $py "$code/mesh_prepare_pose.py" --mesh "$detail/mesh-texture/mesh.glb" --source-view "$detail/mesh-texture/source-view.png" --alpha $alpha --camera $camera --out $poses --blender $blender --render-device OPTIX
if ($LASTEXITCODE -ne 0) { throw '姿态检查失败' }
& $py "$code/mesh_pipeline.py" @common --out 'outputs/轴承座-新检查' --budget 0 --inspection-item 03 --inspection-pose "$poses/pose-45" --inspection-source-view "$detail/mesh-texture/source-view.png"
```

输出包含全图、分层 PSD、局部与结构对照，以及记录接触位置、视角、尺度依据和光色指标的 `result.json`。

仅更新材质时，可用 `mesh_retexture.py` 重新投射素材；它会核对导出前后的顶点和面以及源视图轮廓。`mesh_render_pose.py` 只允许几何完全相同的网格复用已验证落稳姿态。检查入口的 `--inspection-baseline <上一版/result.json>` 固定已验证的接触锚点和尺寸，生成可直接比较的检查图。
