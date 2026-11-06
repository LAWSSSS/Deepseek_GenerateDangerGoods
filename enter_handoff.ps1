$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $PSScriptRoot 'work/huggingface'
Set-Location -LiteralPath $PSScriptRoot
Write-Output "项目：$PSScriptRoot"
Write-Output 'Python：.\work\triposr-env\Scripts\python.exe'
Write-Output 'Photoshop MCP：.\work\photoshop-mcp-env\Scripts\photoshop-mcp-server.exe'
