# PTX Directory

This directory should contain the compiled CUDA kernel file: `optixPathTracer.optixir`

## How to get the compiled kernel:

### Option 1: Copy from sample-projects-solution-working (easiest)
After building the sample-projects-solution-working solution, copy the compiled kernel:

```powershell
Copy-Item "sample-projects-solution-working\lib\ptx\Release\optixPathTracer_generated_optixPathTracer.cu.optixir" "ptx\optixPathTracer.optixir"
```

### Option 2: Compile manually with NVCC
Use the CUDA compiler to compile the `.cu` file to OptiXIR format. See the CMake build scripts in the sample-projects-solution-working for the exact compilation commands.

The compiled kernel is a **build artifact** and should not be committed to git.
