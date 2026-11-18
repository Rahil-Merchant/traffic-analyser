<#
Prepare a Windows GPU box for the three trainings, without starting any training.

Everything lives under -Root, outside the production repo, and nothing global is changed except
installing nothing: Python, uv and all packages are private to -Root. Steps are idempotent; re-run after an
interruption and finished steps are skipped.

    powershell -NoProfile -ExecutionPolicy Bypass -File setup_box.ps1 -Root D:\training
    powershell ... -File setup_box.ps1 -Root D:\training -Steps data,verify

Expects the code at <Root>\code\training (copied from the repo's training/ folder) and, for the data step,
one of: <Root>\data\roboflow.zip, -DataUrl '<Roboflow export link>', or ROBOFLOW_API_KEY in the environment.
Steps: tools, envs, deimv2, weights, data, comet, launchers, verify
#>
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [string[]]$Steps = @('tools', 'envs', 'deimv2', 'weights', 'data', 'comet', 'launchers', 'verify'),
    [string]$DataUrl = '',     # Roboflow export link (app.roboflow.com/ds/...?key=...); holds a key, so pass it, never commit it
    [string]$DeimSha = 'f005f50ce565e63f6c298feccb51901a98c3d035',
    [string]$UvVersion = '0.13.4'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Steps = @($Steps | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })   # -File passes 'a,b' as one string
$known = 'tools', 'envs', 'deimv2', 'weights', 'data', 'comet', 'launchers', 'verify'
$bad = $Steps | Where-Object { $known -notcontains $_ }
if ($bad) { throw "unknown step(s): $($bad -join ', '); known: $($known -join ', ')" }
$Code = Join-Path $Root 'code\training'
$Tools = Join-Path $Root 'tools'
$Envs = Join-Path $Root 'envs'
$Data = Join-Path $Root 'data'
$Weights = Join-Path $Root 'weights'
$Deim = Join-Path $Root 'DEIMv2'
$Uv = Join-Path $Tools 'uv\uv.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
$env:UV_CACHE_DIR = Join-Path $Tools 'uv-cache'        # same drive as envs, so installs are hardlinks
$env:UV_LINK_MODE = 'hardlink'
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path $Root, $Tools, $Envs, $Data, $Weights | Out-Null

function Say($m) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $m" }
function Py($envName) { Join-Path $Envs "$envName\Scripts\python.exe" }
function Run($exe) {
    $rest = $args
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe $($rest -join ' ') failed with exit code $LASTEXITCODE" }
}

# Packages per environment. torch builds come from the PyTorch index through uv's --torch-backend.
$Spec = [ordered]@{
    'yolo26' = @{ backend = 'cu126'; pkgs = @('torch==2.7.1', 'torchvision==0.22.1', 'ultralytics==8.4.208',
            'albumentations>=2.0,<3', 'comet_ml==3.58.10', 'onnx>=1.17,<2', 'onnxslim', 'pyyaml', 'numpy', 'pillow', 'scipy') }
    'rfdetr' = @{ backend = 'cu126'; pkgs = @('torch==2.7.1', 'torchvision==0.22.1', 'rfdetr[train,loggers,onnx]==1.12.0',
            'comet_ml==3.58.10', 'pyyaml') }
    'deimv2' = @{ backend = 'cu124'; pkgs = @('torch==2.5.1', 'torchvision==0.20.1', 'faster-coco-eval>=1.6.7', 'PyYAML',
            'tensorboard', 'scipy', 'calflops', 'transformers', 'comet_ml==3.58.10', 'huggingface_hub', 'safetensors',
            'onnx==1.17.0',     # torch.onnx.export (tools/deployment/export_onnx.py) imports onnx
            'albumentations==2.0.8', 'opencv-python-headless==5.0.0.93') }   # --aug cctv (deim_cctv_aug.py)
}

if ($Steps -contains 'tools') {
    if (-not (Test-Path $Uv)) {
        Say "installing uv $UvVersion into $Tools\uv"
        $zip = Join-Path $Tools 'uv.zip'
        Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        Expand-Archive $zip -DestinationPath (Join-Path $Tools 'uv') -Force
        Remove-Item $zip
    }
    Say "python 3.11 (uv-managed, private to $Tools)"
    Run $Uv python install 3.11 --no-bin
}

if ($Steps -contains 'envs') {
    foreach ($name in $Spec.Keys) {
        $dir = Join-Path $Envs $name
        if (-not (Test-Path (Py $name))) { Say "creating env $name"; Run $Uv venv $dir --python 3.11 --seed }
        Say "installing $name ($($Spec[$name].backend))"
        Run $Uv pip install --python (Py $name) --torch-backend $Spec[$name].backend @($Spec[$name].pkgs)
    }
}

if ($Steps -contains 'deimv2') {
    if (-not (Test-Path (Join-Path $Deim 'train.py'))) {
        Say "fetching DEIMv2 @ $DeimSha"
        $zip = Join-Path $Root 'deimv2.zip'
        Invoke-WebRequest "https://codeload.github.com/Intellindust-AI-Lab/DEIMv2/zip/$DeimSha" -OutFile $zip
        $tmp = Join-Path $Root 'deimv2_unzip'
        Expand-Archive $zip -DestinationPath $tmp -Force
        Move-Item (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName $Deim
        Remove-Item $zip, $tmp -Recurse -Force
        Set-Content (Join-Path $Deim 'COMMIT.txt') $DeimSha
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $Deim 'ckpts') | Out-Null
}

if ($Steps -contains 'weights') {
    Push-Location $Weights
    try {
        Say 'yolo26l.pt'
        Run (Py 'yolo26') -c "from ultralytics.utils.downloads import attempt_download_asset as d; print(d('yolo26l.pt'))"
        Say 'rf-detr-medium.pth'
        Run (Py 'rfdetr') -c "from rfdetr.assets.model_weights import download_pretrain_weights as d; d(r'$Weights\rf-detr-medium.pth')"
        Say 'DEIMv2-L COCO checkpoint'
        Run (Py 'deimv2') (Join-Path $Code 'fetch_deimv2_ckpt.py') --size l --out (Join-Path $Deim 'ckpts\deimv2_dinov3_l_coco.pth')
    } finally { Pop-Location }
}

if ($Steps -contains 'data') {
    $export = Join-Path $Data 'dataset_261110'
    $split = Join-Path $Data 'dataset_261110_split'
    if (-not (Test-Path (Join-Path $export 'data.yaml'))) {
        $zip = Join-Path $Data 'roboflow.zip'
        New-Item -ItemType Directory -Force -Path $export | Out-Null
        if (-not (Test-Path $zip) -and $DataUrl) {
            Say 'downloading the Roboflow export'
            Run curl.exe -L --fail --retry 5 -o $zip $DataUrl
        }
        if (Test-Path $zip) {
            Say "extracting $zip"
            Run tar -xf $zip -C $export
        } elseif ($env:ROBOFLOW_API_KEY) {
            Say 'downloading full-res-labelled v5 (yolov8) from Roboflow'
            Run (Py 'rfdetr') -c "import roboflow; roboflow.Roboflow().workspace('vehicledetection-ioag0').project('full-res-labelled').version(5).download('yolov8', location=r'$export', overwrite=True)"
        } else { throw "no $zip, no -DataUrl and no ROBOFLOW_API_KEY: copy the export zip there first" }
    }
    Say 're-splitting (must reproduce the committed manifest exactly)'
    Push-Location $Code
    try {
        Run (Py 'yolo26') resplit_dataset.py $export --out $split --groups reports\dataset_261110\groups.csv.gz `
            --group-by scene --class-config classes_production.yaml --seed 0 --restarts 400 `
            --expect-manifest reports\dataset_261110\split_manifest.csv.gz
        Run (Py 'yolo26') oversample_rare.py $split
    } finally { Pop-Location }
}

if ($Steps -contains 'comet') {
    $key = [Environment]::GetEnvironmentVariable('COMET_API_KEY', 'User')
    if (-not $key) { $key = [Environment]::GetEnvironmentVariable('COMET_API_KEY', 'Machine') }
    $cfg = Join-Path $env:USERPROFILE '.comet.config'
    if ($key) { Say "COMET_API_KEY is set for this user (length $($key.Length))" }
    elseif (Test-Path $cfg) { Say "using $cfg" }
    else { Write-Warning 'no Comet credentials: set them with  setx COMET_API_KEY <key>  then open a new shell' }
    if ($key) { $env:COMET_API_KEY = $key }
    Run (Py 'yolo26') -c "import comet_ml; api = comet_ml.API(); print('comet ok, workspaces:', api.get_workspaces())"
}

if ($Steps -contains 'launchers') {
    # One launcher per model; every extra argument is passed through to the training script.
    $split = Join-Path $Data 'dataset_261110_split'
    $runs = Join-Path $Root 'runs'
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    $common = @'
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
foreach ($n in 'COMET_API_KEY', 'COMET_WORKSPACE') { $v = [Environment]::GetEnvironmentVariable($n, 'User'); if ($v) { Set-Item "env:$n" $v } }
'@
    Set-Content (Join-Path $Root 'train_yolo26.ps1') ($common + @"

Set-Location '$runs'
& '$(Py 'yolo26')' '$Code\train_yolo26.py' --data '$split\data_rfs.yaml' --pretrained '$Weights\yolo26l.pt' --project '$runs\yolo26' @args
"@)
    Set-Content (Join-Path $Root 'train_rfdetr.ps1') ($common + @"

Set-Location '$runs'
& '$(Py 'rfdetr')' '$Code\train_rfdetr.py' --dataset '$split\coco' --pretrain-weights '$Weights\rf-detr-medium.pth' --output '$runs\rfdetr' @args
"@)
    Set-Content (Join-Path $Root 'train_deimv2.ps1') ($common + @"

& '$(Py 'deimv2')' '$Code\train_deimv2.py' --repo '$Deim' --dataset '$split\coco' --size l --coco-ckpt '$Deim\ckpts\deimv2_dinov3_l_coco.pth' --output '$runs\deimv2' @args
"@)
    Say "launchers: $Root\train_yolo26.ps1, train_rfdetr.ps1, train_deimv2.ps1"
}

if ($Steps -contains 'verify') {
    $probe = "import torch, sys; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); x = torch.randn(256, 256, device='cuda'); print('matmul ok', float((x @ x).abs().mean()) > 0)"
    foreach ($name in $Spec.Keys) { Say "env $name"; Run (Py $name) -c $probe }
    Run (Py 'yolo26') -c "import ultralytics, albumentations, comet_ml; print('ultralytics', ultralytics.__version__, 'albumentations', albumentations.__version__, 'comet_ml', comet_ml.__version__)"
    Run (Py 'rfdetr') -c "import rfdetr, pytorch_lightning, comet_ml; print('rfdetr', rfdetr.__version__ if hasattr(rfdetr, '__version__') else 'ok', 'lightning', pytorch_lightning.__version__)"
    Run (Py 'deimv2') -c "import faster_coco_eval, comet_ml, safetensors; print('deimv2 deps ok')"
    Say 'dry runs (no training)'
    & (Join-Path $Root 'train_yolo26.ps1') --dry-run | Select-Object -Last 3
    & (Join-Path $Root 'train_rfdetr.ps1') --dry-run | Select-Object -First 1
    & (Join-Path $Root 'train_deimv2.ps1') --dry-run | Select-Object -Last 1
    Say 'done'
}
