# Build script to compile CUDA shaders to OptiXIR
# Located in: src/reorganise/cuda/
# Run this after modifying CUDA shader files

# Configuration - Change these paths for your machine if needed
$VCVARS_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$CUDA_PATH_CONFIG = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$OPTIX_INCLUDE = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include"
$OPTIX_SDK = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\SDK"

# Setup Visual Studio environment
if (Test-Path $VCVARS_PATH) {
    cmd /c "`"$VCVARS_PATH`" && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Force -Path "ENV:\$($matches[1])" -Value $matches[2]
        }
    }
    Write-Host "✓ Visual Studio environment loaded" -ForegroundColor Green
}
else {
    Write-Host "✗ Visual Studio vcvars64.bat not found at: $VCVARS_PATH" -ForegroundColor Red
    Write-Host "  Please update the VCVARS_PATH variable at the top of this script" -ForegroundColor Yellow
}

# Setup CUDA paths
$nvccPath = "$CUDA_PATH_CONFIG\bin\nvcc.exe"
$cudaInclude = "$CUDA_PATH_CONFIG\include"

if (-not ((Test-Path $nvccPath) -and (Test-Path $cudaInclude))) {
    Write-Host "✗ CUDA not found at: $CUDA_PATH_CONFIG" -ForegroundColor Red
    Write-Host "  Please update the CUDA_PATH_CONFIG variable at the top of this script" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ CUDA found at: $CUDA_PATH_CONFIG" -ForegroundColor Green

# Setup OptiX paths
$optixInclude = $OPTIX_INCLUDE
$sdkPath = $OPTIX_SDK

if (-not ((Test-Path $optixInclude) -and (Test-Path $sdkPath))) {
    Write-Host "✗ OptiX SDK not found at: $OPTIX_INCLUDE / $OPTIX_SDK" -ForegroundColor Red
    Write-Host "  Please update the OPTIX_INCLUDE and OPTIX_SDK variables at the top of this script" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ OptiX SDK found" -ForegroundColor Green

# File paths (resolve to absolute paths for reliability)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))
$cuFile = Join-Path $projectRoot "src\reorganise\cuda\raytrace.cu"
$photonCuFile = Join-Path $projectRoot "src\reorganise\cuda\photon_emission_combined.cu"
$outputFile = Join-Path $projectRoot "ptx\raytrace.optixir"
$photonOutputFile = Join-Path $projectRoot "ptx\photon_emission.optixir"

Write-Host "Compiling CUDA kernels..." -ForegroundColor Cyan

# Common NVCC arguments
$commonArgs = @(
    "-D__gl_h_", "-D__GL_H__", "-D__X_GL_H", "-DGLAD_GLAPI_EXPORT",
    "-I$optixInclude",
    "-I$cudaInclude",
    "-I$sdkPath\cuda",
    "-I$sdkPath",
    "-I$projectRoot",
    "-I$projectRoot\src",
    "-I$projectRoot\src\reorganise",
    "-I$projectRoot\src\reorganise\cuda",
    "-I$projectRoot\src\reorganise\cuda\photon_emission",
    "--optix-ir",
    "-arch=sm_86",
    "-std=c++11",
    "-use_fast_math",
    "-lineinfo",
    "-rdc=true",
    "-D__x86_64",
    "-D_USE_MATH_DEFINES",
    "-DNOMINMAX"
)

# Compile main raytrace
Write-Host "Compiling raytrace.cu..." -ForegroundColor Gray
& $nvccPath ($commonArgs + @("-o", "$outputFile", "$cuFile"))
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to compile raytrace.cu" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Compile photon emission
Write-Host "Compiling photon_emission_combined.cu..." -ForegroundColor Gray
& $nvccPath ($commonArgs + @("-o", "$photonOutputFile", "$photonCuFile"))
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to compile photon_emission_combined.cu" -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Successfully compiled CUDA kernels:" -ForegroundColor Green
    Write-Host "  - $outputFile" -ForegroundColor Gray
    Write-Host "  - $photonOutputFile" -ForegroundColor Gray
}
else {
    Write-Host "✗ Compilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

