# Builds dist/Subtitula-Setup-<version>.exe. Run from anywhere: powershell -File packaging/build.ps1
param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Step([string]$label, [scriptblock]$action) {
    Write-Host "==> $label"
    & $action
    if ($LASTEXITCODE -ne 0) { throw "$label failed (exit code $LASTEXITCODE)" }
}

$version = (uv run python -c "from whisper_studio import __version__; print(__version__)").Trim()
if (-not $version) { throw "Could not read whisper_studio.__version__" }

$makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
if (-not $makensis) { $makensis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe" }
if (-not (Test-Path $makensis)) { throw "makensis not found. Install NSIS 3 (winget install NSIS.NSIS)." }

if (-not $SkipTests) { Invoke-Step "Tests" { uv run pytest -q } }

Invoke-Step "PyInstaller" {
    uv run pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/whisper_studio.spec
}

Invoke-Step "NSIS installer" {
    & $makensis /V2 "/DVERSION=$version" packaging/installer.nsi
}

$installer = Join-Path $root "dist\Subtitula-Setup-$version.exe"
Write-Host ("Installer: {0} ({1:N0} MB)" -f $installer, ((Get-Item $installer).Length / 1MB))
