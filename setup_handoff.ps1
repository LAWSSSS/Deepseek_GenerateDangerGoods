param(
    [switch]$SkipInstall,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
Set-Location -LiteralPath $project
$env:PYTHONUTF8 = '1'

$pythonArgs = @()
if ($PythonExecutable) {
    $python = (Resolve-Path -LiteralPath $PythonExecutable).Path
} else {
    if (!(Get-Command py -ErrorAction SilentlyContinue)) {
        throw '需要 Windows Python 3.12。可用 -PythonExecutable 指定项目本地解释器。'
    }
    $python = 'py'
    $pythonArgs = @('-3.12')
}
& $python @pythonArgs -c 'import sys; assert sys.version_info[:2] == (3,12)'
if ($LASTEXITCODE -ne 0) { throw '找不到 Python 3.12。' }
& $python @pythonArgs (Join-Path $project 'relocate_handoff.py')
if ($LASTEXITCODE -ne 0) { throw '迁移路径初始化失败。' }

if (!$SkipInstall) {
    $base = Join-Path $project 'work/v2-env'
    $mesh = Join-Path $project 'work/triposr-env'
    $psmcp = Join-Path $project 'work/photoshop-mcp-env'
    foreach ($directory in @($base, $mesh, $psmcp)) {
        if (!(Test-Path -LiteralPath (Join-Path $directory 'Scripts/python.exe'))) {
            & $python @pythonArgs -m venv $directory
            if ($LASTEXITCODE -ne 0) { throw "无法建立虚拟环境：$directory" }
        }
        if ($directory -ne $psmcp) {
            & (Join-Path $directory 'Scripts/python.exe') -c 'import sys; assert sys.version_info[:2] == (3,12)'
            if ($LASTEXITCODE -ne 0) { throw "虚拟环境不是 Python 3.12，请使用新的运行目录：$directory" }
        }
    }
    $basePython = Join-Path $base 'Scripts/python.exe'
    $meshPython = Join-Path $mesh 'Scripts/python.exe'
    $mcpPython = Join-Path $psmcp 'Scripts/python.exe'
    & $basePython -m pip install -r (Join-Path $project 'tools/ps-model-controller/requirements-v2.txt') 'pywin32==312' 'scikit-image==0.26.0' 'scipy==1.18.1'
    if ($LASTEXITCODE -ne 0) { throw '基础依赖安装失败。' }
    Set-Content -LiteralPath (Join-Path $base 'Lib/site-packages/handoff-vendor.pth') -Value '../../../../vendor' -Encoding ascii
    $shared = @(
        '../../../v2-env/Lib/site-packages',
        '../../../v2-env/Lib/site-packages/win32',
        '../../../v2-env/Lib/site-packages/win32/lib',
        '../../../v2-env/Lib/site-packages/pythonwin',
        '../../../../vendor',
        'import pywin32_bootstrap'
    )
    Set-Content -LiteralPath (Join-Path $mesh 'Lib/site-packages/handoff-shared.pth') -Value $shared -Encoding ascii
    & $meshPython -m pip install 'einops==0.8.1' 'omegaconf==2.3.0' 'trimesh==4.6.4' 'psutil==7.1.3' 'fast-simplification==0.1.12' 'pymeshlab==2023.12.post3' 'hf_xet==1.8.0'
    if ($LASTEXITCODE -ne 0) { throw '三维依赖安装失败。' }
    & $meshPython -m pip install 'pydantic-graph==2.68.0'
    if ($LASTEXITCODE -ne 0) { throw '摄影复核依赖安装失败。' }
    $mcpSource = Join-Path $project 'vendor/photoshop-python-api-mcp-server'
    & $mcpPython -m pip install $mcpSource 'photoshop-python-api==0.24.2' 'mcp==1.30.0'
    if ($LASTEXITCODE -ne 0) { throw 'Photoshop MCP 安装失败。' }
    $blenderRoot = Join-Path $project 'work/blender45'
    $blender = Join-Path $blenderRoot 'blender-4.5.3-windows-x64/blender.exe'
    if (!(Test-Path -LiteralPath $blender)) {
        $archive = Join-Path $blenderRoot 'blender-4.5.3-windows-x64.zip'
        if (!(Test-Path -LiteralPath $archive)) { throw '交接包缺少 Blender 压缩包。' }
        Expand-Archive -LiteralPath $archive -DestinationPath $blenderRoot -Force
    }
    & $basePython -m pip check
    if ($LASTEXITCODE -ne 0) { throw '基础依赖检查失败。' }
    & $meshPython -c 'import cv2,torch,trimesh,psutil,pymeshlab,skimage,mobile_sam,mcp,pywintypes; print("Mesh runtime imports OK")'
    if ($LASTEXITCODE -ne 0) { throw '三维环境导入检查失败。' }
}
Write-Output '路径初始化完成。请在当前终端执行：. .\enter_handoff.ps1'
Write-Output 'Photoshop 和有效的 DeepSeek 密钥需要在新设备单独配置。'
