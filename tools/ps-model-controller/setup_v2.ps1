param(
    [Parameter(Mandatory=$true)][string]$RuntimeDirectory,
    [switch]$UseMirrors
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($RuntimeDirectory)
$venv=Join-Path $root 'venv'
$weights=Join-Path $root 'weights'
New-Item -ItemType Directory -Force $root,(Join-Path $weights 'depth'),(Join-Path $weights 'matte') | Out-Null
python -m venv $venv
if($LASTEXITCODE){throw 'Could not create the Python 3.12 environment'}
$python=Join-Path $venv 'Scripts/python.exe'
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-v2.txt')
if($LASTEXITCODE){throw 'Could not install dependencies'}

function Fetch([string]$Target,[string]$Url,[string]$Hash) {
    if((Test-Path -LiteralPath $Target) -and (Get-FileHash -LiteralPath $Target).Hash -eq $Hash){return}
    $temp="$Target.part"
    curl.exe -fLsS --connect-timeout 15 --max-time 900 --retry 1 $Url -o $temp
    if($LASTEXITCODE){throw "Download failed for $([IO.Path]::GetFileName($Target))"}
    if((Get-FileHash -LiteralPath $temp).Hash -ne $Hash){throw "SHA-256 mismatch for $Target"}
    Move-Item -LiteralPath $temp -Destination $Target -Force
}

$commit='f706ad9c4eb7f219c00d9050e46328518ffb65d2'
$codeUrl="https://github.com/ChaoningZhang/MobileSAM/archive/$commit.zip"
$samUrl="https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/$commit/weights/mobile_sam.pt"
$depthUrl='https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f'
$matteUrl='https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx'
if($UseMirrors) {
    $codeUrl="https://gh-proxy.com/$codeUrl"
    $samUrl='https://modelscope.cn/api/v1/models/yatengLG/ISAT_with_segment_anything_checkpoints/repo?Revision=master&FilePath=checkpoints/mobile_sam.pt'
    $depthUrl='https://modelscope.cn/api/v1/models/depth-anything/Depth-Anything-V2-Small-hf/repo?Revision=master&FilePath='
    $matteUrl="https://gh-proxy.com/$matteUrl"
}
# Validate the pinned Git archive revision and use the exact source bundle tested with this runner.
$archive=Join-Path $root 'MobileSAM.zip'
if(!(Test-Path -LiteralPath $archive)) {
    curl.exe -fLsS --connect-timeout 15 --max-time 900 --retry 1 $codeUrl -o "$archive.part"
    if($LASTEXITCODE){throw 'MobileSAM source download failed'}
    Move-Item -LiteralPath "$archive.part" -Destination $archive -Force
}
$check=@'
import hashlib,sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    if archive.comment.decode() != sys.argv[2]:
        raise SystemExit("MobileSAM archive revision mismatch")
    files=[item for item in archive.infolist() if not item.is_dir()]
    if sum(item.file_size for item in files)>150_000_000:
        raise SystemExit("Unexpected MobileSAM archive size")
    h=hashlib.sha256()
    for item in sorted(files,key=lambda i:i.filename.split("/",1)[1]):
        h.update(item.filename.split("/",1)[1].encode())
        h.update(b"\0")
        h.update(hashlib.sha256(archive.read(item)).digest())
    if h.hexdigest()!="c8df6638ad253fc25b827ef40644cbb47dd5a668081d1ce5f691f7d98e894866":
        raise SystemExit("MobileSAM source tree hash mismatch")
'@
$check | & $python - $archive $commit
if($LASTEXITCODE){throw 'MobileSAM archive validation failed'}
& $python -m pip install --no-deps $archive
if($LASTEXITCODE){throw 'MobileSAM install failed'}
Fetch (Join-Path $weights 'mobile_sam.pt') $samUrl '6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F'
$depthFiles=@{
    'config.json'='C56698D3643DDE1F83EA2212759E6B31A22B8F827246A36DD007EE8A22B3FF75'
    'preprocessor_config.json'='D41175C0D889477CA8FC67191E540FAEF14BAF6275157B3FDECF78469E6BBF84'
    'model.safetensors'='3152477CE0D8D6978D76B995120DE97CB5B928701FD0F817769F59E249A16B70'
}
foreach($name in $depthFiles.Keys) {
    $uri=if($UseMirrors){"$depthUrl$name"}else{"$depthUrl/$name"}
    Fetch (Join-Path $weights "depth/$name") $uri $depthFiles[$name]
}
Fetch (Join-Path $weights 'matte/birefnet-general-lite.onnx') $matteUrl '5600024376F572A557870A5EB0AFB1E5961636BEF4E1E22132025467D0F03333'
& $python -m pip check
if($LASTEXITCODE){throw 'Dependency validation failed'}
Write-Output "Python: $python"
Write-Output "Weights: $weights"
