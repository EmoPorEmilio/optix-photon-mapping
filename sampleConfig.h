/*
 * SPDX-FileCopyrightText: Copyright (c) 2009 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// PTX/OptiXIR directory - relative to executable
#define SAMPLES_PTX_DIR "ptx"

// These are used by sutil for runtime compilation (not used in this project)
#define SAMPLES_DIR "."
#define SAMPLES_CUDA_DIR "src/cuda"

// Include directories for runtime compilation (not used - we precompile)
#define SAMPLES_RELATIVE_INCLUDE_DIRS \
  "external/cuda",                    \
      "external/sutil",               \
      "src",                          \
      ".",

#define SAMPLES_ABSOLUTE_INCLUDE_DIRS ""

#define OPTIX_DEBUG_DEVICE_CODE 0
#define OPTIX_OPTIXIR_BUILD_CONFIGURATION "Release"

// We use precompiled OptixIR, not NVRTC runtime compilation
#define CUDA_NVRTC_ENABLED 0

// NVRTC compiler options (kept for compatibility but not used)
#define CUDA_NVRTC_OPTIONS \
  "-std=c++11",            \
      "-arch",             \
      "compute_86",        \
      "-use_fast_math",    \
      "-default-device",   \
      "-rdc",              \
      "true",              \
      "-D__x86_64",

// We generate OptiXIR (not PTX)
#define SAMPLES_INPUT_GENERATE_OPTIXIR 1
#define SAMPLES_INPUT_GENERATE_PTX 0
