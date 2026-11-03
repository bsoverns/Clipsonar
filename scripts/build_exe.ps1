# Builds dist\Clipsonar-windows-cpu.zip. Run from the repository root with the project's
# [cpu] install active, plus: pip install pyinstaller pip-licenses
# The face model and FFmpeg are deliberately left out. See THIRD_PARTY_NOTICES.md.

$ErrorActionPreference = "Stop"

python -m PyInstaller --noconfirm --clean --windowed --name Clipsonar `
    --paths src `
    --collect-all insightface `
    --collect-all onnxruntime `
    --collect-submodules albumentations `
    scripts\gui_entry.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$app = "dist\Clipsonar"
$selfTest = Start-Process "$app\Clipsonar.exe" -ArgumentList "--self-test" -Wait -PassThru
if ($selfTest.ExitCode -ne 0) { throw "Self-test failed with exit code $($selfTest.ExitCode)" }

python -m piplicenses --from=mixed --format=markdown --with-urls --output-file "$app\THIRD_PARTY_LICENSES.md"
if ($LASTEXITCODE -ne 0) { throw "pip-licenses failed" }
Copy-Item LICENSE, README.md, THIRD_PARTY_NOTICES.md $app

Compress-Archive -Path $app -DestinationPath dist\Clipsonar-windows-cpu.zip -Force
Write-Output "Built dist\Clipsonar-windows-cpu.zip"
