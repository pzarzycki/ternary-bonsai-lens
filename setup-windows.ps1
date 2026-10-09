param(
    [ValidateSet('both','cuda','llama')][string]$Backend = 'both',
    [switch]$SkipSync,
    [switch]$SkipWeb,
    [switch]$SkipReadouts
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $SkipSync) {
    uv sync --cache-dir .uv-cache
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
}
if ($Backend -in @('both','cuda')) {
    if (-not (Test-Path models/bonsai2-27b-mlx/model.safetensors)) {
        uv run --no-sync hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir models/bonsai2-27b-mlx
        if ($LASTEXITCODE -ne 0) { throw 'MLX pack download failed' }
    }
}
if ($Backend -in @('both','llama')) {
    $Tag = 'prism-b10754-2459f68'
    $BinDir = Join-Path $PSScriptRoot 'runtime/llama-cuda'
    New-Item -ItemType Directory -Force $BinDir,models/bonsai2-27b-gguf | Out-Null
    $InstalledVersion = ''
    if (Test-Path "$BinDir/llama-server.exe") {
        $InstalledVersion = (& "$BinDir/llama-server.exe" --version 2>&1 | Out-String)
    }
    if ($InstalledVersion -notmatch '2459f68') {
        $Zip = Join-Path $BinDir 'binaries.zip'
        curl.exe -L --fail --retry 3 -o $Zip "https://github.com/PrismML-Eng/llama.cpp/releases/download/$Tag/llama-$Tag-bin-win-cuda-12.4-x64.zip"
        if ($LASTEXITCODE -ne 0) { throw 'Prism binary download failed' }
        Expand-Archive -LiteralPath $Zip -DestinationPath $BinDir -Force
    }
    if (-not (Test-Path "$BinDir/cudart64_12.dll")) {
        $Zip = Join-Path $BinDir 'cudart.zip'
        curl.exe -L --fail --retry 3 -o $Zip "https://github.com/PrismML-Eng/llama.cpp/releases/download/$Tag/cudart-llama-bin-win-cuda-12.4-x64.zip"
        if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime download failed' }
        Expand-Archive -LiteralPath $Zip -DestinationPath $BinDir -Force
    }
    $ModelInfo = Invoke-RestMethod 'https://huggingface.co/api/models/prism-ml/Ternary-Bonsai-2-27B-gguf?blobs=true'
    foreach ($Name in @('Ternary-Bonsai-2-27B-PTQ1_0.gguf','Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf')) {
        $Destination = Join-Path $PSScriptRoot "models/bonsai2-27b-gguf/$Name"
        if (Test-Path -LiteralPath $Destination) { continue }
        $Partial = "$Destination.download"
        curl.exe -L --fail --retry 3 -C - -o $Partial "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/${Name}?download=true"
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name (rerun to resume)" }
        $Expected = ($ModelInfo.siblings | Where-Object rfilename -eq $Name).lfs.sha256
        if (-not $Expected) { throw "Missing official SHA256 metadata: $Name" }
        if ((Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
            throw "SHA256 mismatch: $Name"
        }
        Move-Item -LiteralPath $Partial -Destination $Destination
        Write-Output "Verified $Name"
    }
    if (-not (Test-Path models/bonsai2-27b-mlx/tokenizer.json)) {
        uv run --no-sync hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --include '*.json' '*.jinja' --local-dir models/bonsai2-27b-mlx
        if ($LASTEXITCODE -ne 0) { throw 'Tokenizer download failed' }
    }
}
if ($Backend -in @('both','llama') -and -not $SkipReadouts) {
    $VSWhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
    $Compiler = if (Test-Path $VSWhere) { & $VSWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
    if ($Compiler) {
        & "$PSScriptRoot/native/build-prism-readouts.ps1"
    } else {
        Write-Warning 'Native readouts need Visual Studio C++ Build Tools. Native Chat/Code remain available; auto uses direct CUDA for readouts until the bridge is built.'
    }
}
if (-not $SkipWeb) {
    npm.cmd --prefix web ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    npm.cmd --prefix web run build
    if ($LASTEXITCODE -ne 0) { throw 'Web build failed' }
}
Write-Output 'Setup complete. Run uv run server.py. Auto prefers native Prism with readouts when the bridge is built; BONSAI_BACKEND=cuda keeps the direct implementation.'
