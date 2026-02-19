# WE DO NOT INVENT. WE DO NOT USE KNOWLEDGE FROM OUTSIDE THE PDF. WE ADHERE STRICTLY TO JENSEN'S ALGORITHM AS DESCRIBED IN course8.pdf.

## Codebase Review Plan: Jensen's Photon Mapping

### Code Files to Review
```
src/cuda/photon_emission/    - Photon tracing
src/cuda/photon_gather.h     - Radiance estimate
src/cuda/caustic_lighting/   - Caustic pass
src/cuda/indirect_lighting/  - Global illumination pass
src/rendering/photon/        - KD-tree, photon storage
```

---

## CHUNK 1: Photon Emission & Storage (PDF §1.1-1.3) - VERIFIED

**Jensen's Formulas:**
```
P_photon = P_light / n_e                    (Eq. 1, p.12)
```

**Russian Roulette (PDF p.15-16):**
```
ξ ∈ [0, d]       → diffuse reflection
ξ ∈ ]d, s+d]    → specular reflection
ξ ∈ ]s+d, 1]    → absorption
```
Power adjustment after reflection: `P_refl = P_inc * s / P_s`

**Storage Rules (PDF §1.3.1, p.17):**
- Store ONLY at diffuse (non-specular) surfaces
- DO NOT store at specular surfaces
- Store position, incoming power, incident direction

**Code Verified:**
- `src/cuda/photon_emission/closesthit_store.cu`
- Line 145: Only stores at MATERIAL_DIFFUSE surfaces
- Line 150: `depth > 0` skips direct illumination
- Lines 156-159: `prevWasSpecular` flag correctly routes to caustic vs global map
- Line 173: `throughput *= mat.albedo / mat.diffuseProb` - correct Russian Roulette scaling

---

## CHUNK 2: KD-Tree (PDF §2) - VERIFIED

**Jensen's Structure (PDF p.23):**
- Balanced kd-tree, heap-like: element i has children 2i, 2i+1
- Split dimension = axis with largest extent
- Median split

**Code Verified:**
- `src/rendering/photon/PhotonKDTree.h`
- Uses recursive median split with `std::nth_element`
- Cycles axis with `depth % 3`
- Uploaded to GPU as device-friendly linear array

---

## CHUNK 3: Radiance Estimate (PDF §3.1, Eq. 8) - VERIFIED

**Jensen's Formula (PDF p.26):**
```
L_r(x,ω) ≈ (1/πr²) Σ f_r(x,ω_p,ω) * ΔΦ_p(x,ω_p)
```
- r = radius of sphere containing N photons
- ΔA = πr² (projected disk area)

**Cone Filter (PDF §3.2.1, Eq. 10-11):**
```
w_pc = 1 - d_p/(k*r)     where k ≥ 1

L_r(x,ω) = Σ[f_r * ΔΦ_p * w_pc] / ((1 - 2/3k) * πr²)
```

**Code Verified:**
- `src/cuda/photon_gather.h`
- Line 60: Cone filter weight `1 - sqrt(dist_sq) / gather_radius`
- Line 67: Area normalization `PM_PI * radius_sq`
- Line 11: `PM_CONE_FILTER_K = 3.0f`
- Lines 106-110: Incident direction check `incidentDot < 0` (photon from outside)

---

## CHUNK 4: Caustics (PDF §1.5, §4.3) - BUG FIXED

**Jensen's Definition (PDF p.21):**
```
Caustic photon map: LS+D
- L = light emission
- S+ = one or more specular reflections/transmissions
- D = diffuse surface (where stored)
```

**Rendering (PDF §4.3, p.37):**
- Use radiance estimate from caustic photon map DIRECTLY
- Never use Monte Carlo for caustics

### BUG FOUND AND FIXED:
**Problem:** `closesthit_caustic.cu` was not multiplying by surface albedo (BRDF f_r term)

**Jensen's Eq. 8 requires:** `L_r ≈ (1/πr²) Σ f_r * ΔΦ_p`

**Before (incorrect):**
```cpp
float3 caustics = gatherPhotonsRadiance(...);  // Missing f_r!
```

**After (correct):**
```cpp
Material mat = params.triangle_materials[prim_idx];
float3 caustics = gatherPhotonsRadianceWithAlbedo(..., mat.albedo, ...);
```

**Files Modified:**
1. `src/cuda/caustic_lighting/caustic_launch_params.h` - Added `Material* triangle_materials`
2. `src/cuda/caustic_lighting/closesthit_caustic.cu` - Added material lookup and albedo multiplication
3. `src/optix/OptixManager.cpp` line 1140 - Pass `triangle_materials` to caustic params

---

## CHUNK 5: Global/Indirect (PDF §4.4) - VERIFIED

**Jensen's Definition (PDF p.21):**
```
Global photon map: L{S|D|V}*D
```

**Rendering (PDF §4.4, p.38):**
- Indirect = light reflected diffusely at least once
- Use final gather for smooth results (optional)

**Code Verified:**
- `src/cuda/indirect_lighting/closesthit_indirect.cu`
- Line 42: Correctly uses `gatherPhotonsRadianceWithAlbedo` with surface albedo
- Uses global photon map (not caustic) - correct separation

---

## CHUNK 6: CUDA Optimization - PENDING

**Known bottleneck:** KD-tree CPU build (184ms = 11%)
**PDF allows:** GPU-based approaches not in original paper

**Opportunities:**
1. GPU KD-tree construction
2. Parallel radix sort for photon balancing
3. Warp-level photon gathering optimization

---

## CHUNK 7: Fog/Participating Media (PDF §1.4, §3.3) - IMPLEMENTED

**Jensen's Formula (PDF Eq. 14):**
```
Volume radiance: L_i ≈ (1/σ_t) Σ f(x,ω'_p,ω) * ΔΦ_p / (4/3 πr³)
```

**Implementation Details:**

### Files Created:
1. `src/rendering/photon/VolumePhoton.h` - Volume photon structure
   - `VolumePhoton`: position, power, direction (for phase function)
   - `VolumeProperties`: sigma_t, sigma_s, bounds_min/max, albedo()

2. `src/cuda/volume_photon_gather.h` - Jensen's Eq. 14 implementation
   - `gatherVolumePhotonsLinear()`: Linear O(n) volume gathering
   - `gatherVolumeRadianceAlongRay()`: Ray marching through volume
   - `intersectVolumeBounds()`: AABB intersection for volume bounds
   - `computeVolumeTransmittance()`: Beer-Lambert law

### Files Modified:
1. `src/cuda/photon_emission/photon_launch_params.h`
   - Added: enable_volume_scattering, volume, volume_photons_out, volume_photon_counter

2. `src/cuda/photon_emission/raygen_photons.cu`
   - Volume scattering in bounce loop: sample t = -ln(ξ)/σ_t
   - Russian Roulette with albedo = σ_s/σ_t
   - Store volume photon at scatter position
   - Isotropic phase function (uniform sphere direction)

3. `src/rendering/photon/PhotonTrajectory.h`
   - Added: EVENT_VOLUME_SCATTER, EVENT_VOLUME_ABSORBED, TRAJ_MAT_VOLUME

4. `src/optix/OptixManager.h/cpp`
   - Volume buffer allocation (d_volume_photon_buffer, d_volume_photon_counter)
   - enableVolumeScattering(), setVolumeProperties()
   - Pass volume data to photon emission and direct lighting passes

5. `src/cuda/direct_lighting/direct_launch_params.h`
   - Added fog parameters: enable_fog, volume, volume_photons, fog_color

6. `src/cuda/direct_lighting/raygen_direct.cu`
   - applyFog(): Beer-Lambert transmittance + in-scattering
   - Fog applied after surface hit

7. `src/core/Application.cpp`
   - Enable fog with configurable parameters
   - Ground fog: lower half (y < 274) with σ_t=0.005, σ_s=0.004

**Parameters:**
```cpp
VolumeProperties fogProps;
fogProps.sigma_t = 0.005f;  // Extinction coefficient
fogProps.sigma_s = 0.004f;  // Scattering coefficient (albedo ~0.8)
fogProps.bounds_min = make_float3(0, 0, 0);
fogProps.bounds_max = make_float3(556, 274, 559);  // Ground fog
```

**Verified:** 10455 volume photons stored during test run.

---

## CHUNK 8: Future Features (PDF §5.8) - PENDING

**Marble/Subsurface (PDF §5.8):**
- Volume photon map inside translucent material
- Multiple scattering via photon map
- Requires material boundaries and subsurface transport
