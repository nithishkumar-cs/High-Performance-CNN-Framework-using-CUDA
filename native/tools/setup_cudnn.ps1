# Stages a local cuDNN install under third_party/cudnn/ for CMake's DETECTOR_CUDNN
# option, using the pip-installed nvidia-cudnn-cu12 wheel instead of the full NVIDIA
# Developer Program SDK download (no NVIDIA account needed).
#
# The Windows wheel ships cudnn.h/cudnn*.dll but, unlike the official SDK, no import
# library (.lib) to link against at build time - MSVC needs one. This script generates
# one from the DLL's export table (dumpbin /exports -> a generated .def -> `lib /def`),
# a standard, lossless technique for linking against a DLL that has no .lib (not a
# correctness compromise: the resulting cudnn.lib just re-exposes the DLL's real symbols).
#
# Usage (from a normal PowerShell prompt, not needed inside an agent session re-run):
#   pip install nvidia-cudnn-cu12
#   powershell -ExecutionPolicy Bypass -File native/tools/setup_cudnn.ps1
# Then configure CMake with -DDETECTOR_CUDA=ON -DDETECTOR_CUDNN=ON (CUDNN_ROOT defaults
# to third_party/cudnn, overridable via -DCUDNN_ROOT=<path> to use an official SDK instead).

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$dest = Join-Path $repoRoot "third_party\cudnn"
New-Item -ItemType Directory -Force -Path "$dest\include", "$dest\lib", "$dest\bin" | Out-Null

$cudnnPkgRoot = (python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))").Trim()
if (-not (Test-Path $cudnnPkgRoot)) {
    throw "nvidia-cudnn-cu12 is not installed (pip install nvidia-cudnn-cu12 first)"
}

Copy-Item "$cudnnPkgRoot\include\*.h" "$dest\include\" -Force
Copy-Item "$cudnnPkgRoot\bin\*.dll" "$dest\bin\" -Force

$dll = Get-ChildItem "$cudnnPkgRoot\bin\cudnn64_*.dll" | Select-Object -First 1
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$work = Join-Path $env:TEMP "cudnn_import_lib"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$defPath = Join-Path $work "cudnn.def"
$libBase = [System.IO.Path]::GetFileNameWithoutExtension($dll.Name)

$bat = Join-Path $work "gen.bat"
@"
@echo off
call "$vcvars"
dumpbin /exports "$($dll.FullName)" > "$work\exports.txt"
"@ | Set-Content -Path $bat -Encoding ascii
cmd.exe /c $bat | Out-Null

$names = Select-String -Path "$work\exports.txt" -Pattern '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\w+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value }
"LIBRARY $libBase" | Set-Content -Path $defPath -Encoding ascii
"EXPORTS" | Add-Content -Path $defPath -Encoding ascii
$names | ForEach-Object { "  $_" } | Add-Content -Path $defPath -Encoding ascii

$bat2 = Join-Path $work "gen2.bat"
@"
@echo off
call "$vcvars"
cd "$work"
lib /def:"$defPath" /out:cudnn.lib /machine:x64
"@ | Set-Content -Path $bat2 -Encoding ascii
cmd.exe /c $bat2 | Out-Null

Copy-Item "$work\cudnn.lib" "$dest\lib\cudnn.lib" -Force
Write-Host "Staged cuDNN (version dir: $cudnnPkgRoot) under $dest"
Write-Host "$($names.Count) exported symbols captured in the generated import library."
