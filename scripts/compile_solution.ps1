param(
    [switch]$SkipCuda,
    [switch]$SkipBuild,
    [switch]$Release
)

# ============================================================================
# OptiX Photon Mapping - Build Script
# ============================================================================
# Compiles CUDA shaders and builds the Visual Studio solution.
#
# Usage:
#   .\compile_solution.ps1              # Build Debug
#   .\compile_solution.ps1 -Release     # Build Release
#   .\compile_solution.ps1 -SkipCuda    # Skip CUDA compilation
# ============================================================================

# ==========================
# CONFIGURATION - Edit these paths for your system
# ==========================
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$OPTIX_PATH = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"
$OPTIX_SAMPLES_BUILD = "C:\Users\emili\Desktop\proyectos\optix_samples_build"

# ==========================

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir

Write-Host "=== OptiX Photon Mapping - Build ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "CUDA:   $CUDA_PATH" -ForegroundColor Gray
Write-Host "OptiX:  $OPTIX_PATH" -ForegroundColor Gray
Write-Host "Build:  $OPTIX_SAMPLES_BUILD" -ForegroundColor Gray
Write-Host ""

# Validate paths
if (-not (Test-Path $CUDA_PATH)) {
    Write-Host "ERROR: CUDA not found at $CUDA_PATH" -ForegroundColor Red
    Write-Host "Edit the CUDA_PATH variable at the top of this script" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $OPTIX_PATH)) {
    Write-Host "ERROR: OptiX not found at $OPTIX_PATH" -ForegroundColor Red
    Write-Host "Edit the OPTIX_PATH variable at the top of this script" -ForegroundColor Yellow
    exit 1
}

# Detect Visual Studio
$vsPath = ""
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
)
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vsPath = $path
        break
    }
}

if (-not $vsPath) {
    Write-Host "ERROR: Visual Studio 2022 not found" -ForegroundColor Red
    exit 1
}

$msbuildPath = "$vsPath\MSBuild\Current\Bin\MSBuild.exe"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

# ============================================================================
# Step 1: Compile CUDA shaders
# ============================================================================

if (-not $SkipCuda) {
    Write-Host "Step 1: Compiling CUDA shaders..." -ForegroundColor Yellow
    
    # Setup Visual Studio environment
    if (Test-Path $vcvarsPath) {
        cmd /c "`"$vcvarsPath`" && set" | ForEach-Object {
            if ($_ -match "^(.*?)=(.*)$") {
                Set-Item -Force -Path "ENV:\$($matches[1])" -Value $matches[2]
            }
        }
    }
    
    $nvccPath = "$CUDA_PATH\bin\nvcc.exe"
    $optixInclude = "$OPTIX_PATH\include"
    $optixSdk = "$OPTIX_PATH\SDK"
    
    # Common NVCC arguments
    $commonArgs = @(
        "-D__gl_h_", "-D__GL_H__", "-D__X_GL_H", "-DGLAD_GLAPI_EXPORT",
        "-I$optixInclude",
        "-I$CUDA_PATH\include",
        "-I$optixSdk\cuda",
        "-I$optixSdk\sutil",
        "-I$optixSdk",
        "-I$projectRoot",
        "-I$projectRoot\src",
        "-I$projectRoot\src\cuda",
        "-I$projectRoot\src\cuda\photon_emission",
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
    
    $ptxDir = Join-Path $projectRoot "ptx"
    $cudaDir = Join-Path $projectRoot "src\cuda"
    
    # List of shaders to compile
    $shaders = @(
        @{ src = "raytrace.cu"; out = "raytrace.optixir"; extra = @() },
        @{ src = "photon_emission_combined.cu"; out = "photon_emission.optixir"; extra = @() },
        @{ src = "direct_lighting_combined.cu"; out = "direct_lighting.optixir"; extra = @("-I$cudaDir\direct_lighting") },
        @{ src = "indirect_lighting_combined.cu"; out = "indirect_lighting.optixir"; extra = @("-I$cudaDir\indirect_lighting") },
        @{ src = "caustic_lighting_combined.cu"; out = "caustic_lighting.optixir"; extra = @("-I$cudaDir\caustic_lighting") },
        @{ src = "specular_lighting_combined.cu"; out = "specular_lighting.optixir"; extra = @("-I$cudaDir\specular_lighting") }
    )
    
    foreach ($shader in $shaders) {
        $srcFile = Join-Path $cudaDir $shader.src
        $outFile = Join-Path $ptxDir $shader.out
        
        Write-Host "  Compiling $($shader.src)..." -ForegroundColor Gray
        
        $args = $commonArgs + $shader.extra + @("-o", $outFile, $srcFile)
        
        & $nvccPath $args
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $($shader.src)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
    
    Write-Host "CUDA shaders compiled" -ForegroundColor Green
    Write-Host ""
}

# ============================================================================
# Step 2: Build Visual Studio solution
# ============================================================================

if (-not $SkipBuild) {
    $config = if ($Release) { "Release" } else { "Debug" }
    Write-Host "Step 2: Building solution ($config)..." -ForegroundColor Yellow
    
    $vcxproj = Join-Path $projectRoot "OptixPhotonMappingRayTracing.vcxproj"
    
    & $msbuildPath $vcxproj `
        /p:Configuration=$config `
        /p:Platform=x64 `
        /p:CUDA_PATH="$CUDA_PATH" `
        /p:OPTIX_PATH="$OPTIX_PATH" `
        /p:OPTIX_SAMPLES_BUILD="$OPTIX_SAMPLES_BUILD" `
        /v:minimal
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful" -ForegroundColor Green
        Write-Host ""
        Write-Host "Run: .\bin\$config\OptixPhotonMappingRayTracing.exe" -ForegroundColor Cyan
    }
    else {
        Write-Host "Build failed" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
