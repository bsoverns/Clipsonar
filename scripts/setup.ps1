# One-time setup from source. Run from the repository root:
#   .\scripts\setup.ps1          (NVIDIA GPU)
#   .\scripts\setup.ps1 -Cpu     (no GPU)

param([switch]$Cpu)

$ErrorActionPreference = "Stop"

# Pinned because insightface and onnxruntime can lag behind new Python releases.
& py -3.13 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.13 is required. Install it with: winget install Python.Python.3.13" }

if (-not (Test-Path ".venv")) { py -3.13 -m venv .venv }

$extra = if ($Cpu) { "cpu" } else { "gpu" }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[$extra]"
if ($LASTEXITCODE -ne 0) {
    throw "Install failed. InsightFace compiles a small extension, which needs Microsoft C++ Build Tools: winget install Microsoft.VisualStudio.2022.BuildTools --override '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive'"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "FFmpeg was not found. Install it with 'winget install ffmpeg', then open a new terminal."
}

Write-Output "Setup complete. Start Clipsonar with:  .\.venv\Scripts\clipsonar-gui.exe"
