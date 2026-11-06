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
