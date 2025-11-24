# Configuration - Change these paths for your machine if needed
$MSBUILD_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"

# Script parameters (set to $true to skip steps)
$SkipCuda = $false
$SkipBuild = $false

Write-Host "=== OptiX Raytracing Solution Compiler ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Compile CUDA shaders (unless skipped)
if (-not $SkipCuda) {
    Write-Host "Step 1: Compiling CUDA shaders..." -ForegroundColor Yellow
    Push-Location "src\cuda"
    & ".\compile_cuda_files.ps1"
    Pop-Location

    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA compilation failed! Aborting." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "CUDA compilation successful" -ForegroundColor Green
    Write-Host ""
}

# Step 2: Build the solution (unless skipped)
if (-not $SkipBuild) {
    Write-Host "Step 2: Building Visual Studio solution..." -ForegroundColor Yellow

    if (-not (Test-Path $MSBUILD_PATH)) {
        Write-Host "MSBuild not found at: $MSBUILD_PATH" -ForegroundColor Red
        Write-Host "Please update the MSBUILD_PATH variable at the top of this script" -ForegroundColor Yellow
        exit 1
    }

    # MSBuild needs to be run from the project directory
    $projectRoot = Split-Path -Parent $PSScriptRoot
    & $MSBUILD_PATH "$projectRoot\OptixPhotonMappingRayTracing.vcxproj" /p:Configuration=Debug /p:Platform=x64 /v:q

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Solution build successful" -ForegroundColor Green
        Write-Host ""
        Write-Host "Ready to run! Execute: .\bin\Debug\OptixPhotonMappingRayTracing.exe" -ForegroundColor Green
    }
    else {
        Write-Host "Solution build failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "=== Build Complete ===" -ForegroundColor Cyan
