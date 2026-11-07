# 废钢监控异物合成

从完整迁移包恢复的 Windows 项目。代码在 `outputs/ps-model-controller`；原图、模型、历史结果和虚拟环境保存在本机工作区，Git 只跟踪代码、配置示例和运行说明。查看 [控制器说明](outputs/ps-model-controller/README.md)。

当前试产范围是 **03 轴承座、10 千斤顶、27 减震器、30 液压缸**。只有四件尺寸与自然度都获得确认后才扩批。三维结构、承托、净空、源图可见面和视觉验收的门槛独立生效。

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
