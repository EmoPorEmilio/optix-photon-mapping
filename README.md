# OptiX Photon Mapping Ray Tracer

A GPU-accelerated photon mapping renderer implementing Henrik Jensen's algorithm
using NVIDIA OptiX 9.0.

## Features

- **Photon Mapping**: Global illumination via photon tracing and radiance
  estimation
- **Caustics**: Light focusing through glass and reflective surfaces
- **Multiple Render Modes**: Direct, indirect, caustic, specular, and combined
- **Interactive Camera**: Orbit, pan, zoom controls
- **XML Configuration**: Customize scene, materials, and rendering parameters

## Prerequisites

1. **NVIDIA CUDA Toolkit** (v13.0 or compatible)
   - https://developer.nvidia.com/cuda-downloads

2. **NVIDIA OptiX SDK** (9.0.0)
   - https://developer.nvidia.com/designworks/optix/download

3. **Visual Studio 2022** with C++ workload

4. **OptiX SDK Samples** (built with CMake)
   ```powershell
   cd "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\SDK"
   cmake -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Debug
   cmake --build build --config Release
   ```

## Build

1. Edit paths at the top of `scripts/compile_solution.ps1` if your installations
   differ:
   ```powershell
   $CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
   $OPTIX_PATH = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"
   $OPTIX_SAMPLES_BUILD = "C:\path\to\your\optix_samples_build"
   ```

2. Build:
   ```powershell
   .\scripts\compile_solution.ps1
   ```

3. Run:
   ```powershell
   .\bin\Debug\OptixPhotonMappingRayTracing.exe
   ```

## Configuration

Edit `bin/Debug/configuration.xml` to customize the scene.

## Controls

| Key            | Action                        |
| -------------- | ----------------------------- |
| **M**          | Cycle render modes            |
| **E**          | Export combined render to PPM |
| **ESC**        | Exit                          |
| **Left drag**  | Orbit camera                  |
| **Right drag** | Pan camera                    |
| **Scroll**     | Zoom                          |

## Render Modes (press M to cycle)

1. **Global Photons** - Visualize photon map as dots
2. **Caustic Photons** - Visualize caustic photon map
3. **Direct Lighting** - Shadow rays only
4. **Indirect Lighting** - Color bleeding from photon map
5. **Caustic Lighting** - Focused light patterns
6. **Specular Lighting** - Reflections and refractions
7. **Combined** - All modes blended with weights

## Project Structure

```
optix-photon-mapping/
├── src/
│   ├── core/          # Application, config loader
│   ├── cuda/          # GPU shaders (raygen, closesthit, miss)
│   ├── optix/         # OptiX pipeline management
│   ├── rendering/     # Render mode implementations
│   ├── scene/         # Camera, objects, materials
│   └── ui/            # Window, input handling
├── ptx/               # Compiled CUDA shaders (.optixir)
├── scripts/           # Build scripts
├── assets/            # 3D models
└── bin/Debug/         # Executable and configuration
```

## License

BSD-3-Clause
