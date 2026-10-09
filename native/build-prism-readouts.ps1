param([string]$Source = "$PSScriptRoot/../runtime/prism-source",
      [string]$Binary = "$PSScriptRoot/../runtime/llama-cuda")
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Tag = 'prism-b10754-2459f68'
if (-not (Test-Path "$Source/include/llama.h")) {
    git clone --depth 1 --branch $Tag https://github.com/PrismML-Eng/llama.cpp.git $Source
    if ($LASTEXITCODE) { throw 'Prism header checkout failed' }
}
$Commit = git -C $Source rev-parse HEAD
if ($Commit -ne '2459f68b5c0eb26261fd5a81682004b93cd645ba') { throw 'Prism headers must match pinned binary' }
$Version = & "$Binary/llama-server.exe" --version 2>&1 | Out-String
if ($Version -notmatch '2459f68') { throw 'Prism binaries must match pinned headers' }
$VS = & "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VS) { throw 'Install Visual Studio Build Tools with Desktop development with C++' }
Import-Module "$VS/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $VS -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
$Build = Join-Path $Root 'runtime/prism-bridge-build'
New-Item -ItemType Directory -Force $Build | Out-Null
$Libraries = @()
foreach ($Name in @('llama','ggml','ggml-base','mtmd')) {
    $Exports = & dumpbin /nologo /exports "$Binary/$Name.dll"
    $Symbols = @($Exports | ForEach-Object {
        if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)') { $Matches[1] }
    })
    if (-not $Symbols.Count) { throw "Cannot read $Name exports" }
    @("LIBRARY $Name.dll",'EXPORTS') + $Symbols | Set-Content "$Build/$Name.def"
    & lib /nologo /machine:x64 "/def:$Build/$Name.def" "/out:$Build/$Name.lib" | Out-Null
    if ($LASTEXITCODE) { throw "Import library failed: $Name" }
    $Libraries += "$Build/$Name.lib"
}
& cl /nologo /LD /O2 /EHsc /std:c++17 /MD "/I$Source/include" "/I$Source/src" "/I$Source/ggml/include" "/I$Source/tools/mtmd" "$PSScriptRoot/prism_readouts.cpp" "/Fo$Build/prism_readouts.obj" /link $Libraries "/OUT:$Binary/bonsai-readouts.dll" "/IMPLIB:$Build/bonsai-readouts.lib"
if ($LASTEXITCODE) { throw 'Readout bridge build failed' }
Write-Output "Built $Binary/bonsai-readouts.dll against $Tag. Existing Prism CUDA kernels are unchanged."
