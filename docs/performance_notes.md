# Performance Notes - Photon Mapping Implementation

## Test Environment
- **GPU**: NVIDIA RTX (SM 8.6 architecture)
- **CUDA**: v13.0
- **OptiX**: 7.x with disk cache enabled
- **Scene**: Cornell Box (556x548x559 units), 12 triangles, quad light on ceiling

---

## Photon Tracing Performance Comparison

### CPU Implementation (Instant Mode)
- **Method**: Sequential ray-surface intersection tests using `CollisionDetector`
- **Collision Detection**: Analytical wall/sphere tests + Möller-Trumbore for triangles

| Photons Launched | Photons Stored | Time (ms) | Rate (photons/sec) |
|------------------|----------------|-----------|-------------------|
| 1,000            | 2,303          | 14.6      | 68,437            |
| 100,000          | ~225,000       | 1,364     | 73,306            |

**Observations:**
- CPU rate relatively stable at ~70K photons/sec
- Stores ~2.2x photons vs launched (multiple bounces per photon)
- Russian Roulette with 70% survival probability

### OptiX GPU Implementation (Instant Mode)
- **Method**: GPU ray tracing with OptiX 7 pipelines
- **Acceleration Structure**: BVH over triangle geometry (IAS with single triangle GAS)

| Photons Launched | Photons Stored | Time (ms) | Rate (photons/sec) |
|------------------|----------------|-----------|-------------------|
| 100,000          | 33,006         | 0.92      | 108,897,000       |

**Observations:**
- GPU rate: ~109M photons/sec
- **Speedup: ~1,500x faster than CPU**
- First launch includes JIT compilation (~3 sec with cache miss)
- Subsequent launches use cached PTX (~1ms)
- Stores fewer photons due to 50% survival probability in GPU shader (vs 70% CPU)

---

## OptiX Pipeline Statistics

### Photon Emission Pipeline
```
Modules: 1
Entry functions: 5 (raygen, miss, closesthit_triangle, closesthit_sphere, intersection_sphere)
Trace calls: 1 per bounce iteration
Register count: 124 per function
```

### Shader Complexity
- `__raygen__photon_emitter`: 255 instructions, 7 basic blocks
- `__closesthit__photon_hit`: 264 instructions, 13 basic blocks
- `__closesthit__photon_sphere_hit`: 245 instructions, 16 basic blocks
- `__intersection__photon_sphere`: 103 instructions, 8 basic blocks

---

## Memory Usage

### Per-Photon Storage
```cpp
struct Photon {
    float3 position;    // 12 bytes
    float3 power;       // 12 bytes
    float3 incidentDir; // 12 bytes
    short flag;         // 2 bytes
    // Total: 38 bytes (aligned to 40 bytes)
};
```

### Buffer Allocations
- Photon buffer: `num_photons * sizeof(Photon)` = 100K * 40 = 4 MB
- Photon counter: 4 bytes (atomic counter)
- Triangle materials: `num_triangles * sizeof(Material)` = 12 * 32 = 384 bytes

---

## Key Findings

### GPU Advantages
1. **Massive parallelism**: Each photon traced independently on separate GPU thread
2. **Hardware RT cores**: BVH traversal accelerated by dedicated silicon
3. **Disk caching**: OptiX caches compiled PTX, eliminating JIT overhead on subsequent runs

### CPU Advantages
1. **Debugging**: Animated mode shows photon paths visually
2. **Flexibility**: Easy to modify collision detection logic
3. **No GPU dependency**: Works on any system

### Bottlenecks Identified
1. **CPU**: Sequential processing, ~14ms for 1K photons
2. **GPU first launch**: JIT compilation ~3 seconds (cached after)
3. **Memory transfer**: `cudaMemcpy` for photon data GPU→CPU (~negligible for 33K photons)

---

## Configuration Options

```json
{
  "photon_mapping_params": {
    "max_photons": 100000,
    "animated": {
      "enabled": false,    // true = CPU animated, false = GPU instant
      "photon_speed": 200.0,
      "emission_interval": 0.5
    }
  }
}
```

---

## Recommendations

1. **Production**: Use OptiX GPU mode with 1M+ photons
2. **Development/Debug**: Use animated CPU mode with 10-100 photons
3. **Survival Probability**: Consider matching CPU (70%) and GPU (50%) for consistency
4. **Photon Budget**: GPU can handle 10M+ photons in <100ms

---

## Future Optimizations

- [ ] Increase GPU survival probability to 70% for more stored photons
- [ ] Implement kd-tree construction on GPU
- [ ] Add photon visualization in GPU mode (splat rendering)
- [ ] Profile memory bandwidth for large photon counts
- [ ] Test with complex scenes (Stanford models, caustic geometry)


