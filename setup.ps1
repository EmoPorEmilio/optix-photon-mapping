# Setup script for OptiX PathTracer
# This script copies the compiled CUDA kernel from the sample project build

Write-Host "OptiX PathTracer Setup Script" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Check if sample project PTX exists
$sourcePath = "C:\Users\emili\Desktop\proyectos\optix_samples_build\lib\ptx\Release\optixPathTracer_generated_optixPathTracer.cu.optixir"
$destPath = "ptx\optixPathTracer.optixir"

if (Test-Path $sourcePath) {
    Write-Host "✓ Found compiled kernel at: $sourcePath" -ForegroundColor Green
    
    # Create ptx directory if it doesn't exist
    if (!(Test-Path "ptx")) {
        New-Item -ItemType Directory -Path "ptx" | Out-Null
        Write-Host "✓ Created ptx directory" -ForegroundColor Green
    }
    
    # Copy the file
    Copy-Item $sourcePath $destPath -Force
    Write-Host "✓ Copied kernel to: $destPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Setup complete! You can now build optixPathTracer.sln in Visual Studio." -ForegroundColor Green
} else {
    Write-Host "✗ Compiled kernel not found at: $sourcePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build the OptiX samples first using CMake:" -ForegroundColor Yellow
    Write-Host "  cd sample-projects-solution-working" -ForegroundColor Yellow
    Write-Host "  cmake --build . --config Release" -ForegroundColor Yellow
    Write-Host "  (Then run this script again)" -ForegroundColor Yellow
    exit 1
}
