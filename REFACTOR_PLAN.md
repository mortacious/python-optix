# python-optix 2.0 Refactor Plan

Status: accepted plan, work not started
Baseline: branch `feature/optix_9_0_0_support` (OptiX 9.0.0, Cython + CuPy bindings)
Target: python-optix 2.0 - pure-Python ctypes host bindings + NVIDIA Warp based Python kernel authoring

## 1. Goals

1. **Python-authored ray tracing kernels.** Users write OptiX programs (ray-gen, closest-hit,
   miss, intersection, callables) as Python functions using Warp decorators. The library
   compiles them via Warp's NVRTC toolchain to PTX and loads them with `optixModuleCreate`.
   Raw CUDA C++ / PTX / OptixIR pass-through remains as an escape hatch.
2. **Pure-Python host bindings.** Replace Cython with ctypes. No compiler, no Cython, no CUDA
   Toolkit and no OptiX SDK required at install time. Ship a universal `py3-none-any` wheel.
3. **Tiered API.** A faithful low-level layer, a refactored pythonic layer with automatic
   memory management and sane defaults, and an optional high-level scene API - with escape
   hatches from every tier down to the one below.
4. **Sustainable maintenance.** Multi-version OptiX support via per-version declaration data
   instead of per-version recompilation; a real test suite; a working CI.

## 2. Verified technical foundations

These facts were verified against primary sources (OptiX headers and programming-guide
material, Warp 1.17 source and design docs, NVIDIA samples and forums). They are the load-
bearing assumptions of this plan; if any of them breaks, the affected phase must be
re-examined.

- **F1 - OptiX host API needs no linking.** Since OptiX 7 the host API is loaded by
  `dlopen("libnvoptix.so.1")` (Linux) / `LoadLibrary("nvoptix.dll")` (Windows, both shipped
  with the display driver), then `dlsym("optixQueryFunctionTable")` fills an
  `OptixFunctionTable`. `optix_stubs.h` and `optix_function_table_definition.h` (which define
  this mechanism) are BSD-3-Clause licensed. A ctypes binding therefore needs no OptiX
  headers at build time and none at runtime for the host layer. NVIDIA's own
  `nvidia-ml-py` (pure-Python ctypes over the driver-shipped NVML) and the OptiX_Apps
  `intro_driver` sample validate the pattern.
- **F2 - OptiX accepts PTX with unresolved extern symbols.** The OptiX device intrinsics
  (`optixTrace`, `optixGetLaunchIndex`, `optixGetSbtDataPointer`, ...) are inline wrappers
  around inline-PTX calls to undefined `_optix_*` symbols (`_optix_trace_typed_32`,
  `_optix_get_sbt_data_ptr_64`, ...) that the OptiX runtime resolves when it compiles the
  module. Compiling user code with NVRTC to PTX containing such extern calls is the official
  flow (the OptiX SDK itself uses NVRTC with `-rdc true`). ptxas rejects such PTX; OptiX
  accepts it. CUBIN is not accepted - module input must be PTX or OptixIR.
- **F3 - Warp 1.17 emits OptiX-shaped programs.** Warp 1.17 (Aug 2026) added external
  compilation extensions (GH issue #1575, design doc
  `design/external-compilation-extensions.md`, which names OptiX as the first consumer):
  - `@wp.kernel(name="__raygen__rg", entry_point_abi="external_constant_params",
    module_options={"strip_hash": True}, enable_backward=False)` generates exactly
    `extern "C" __global__ void __raygen__rg()` plus a `__constant__` params global - no
    arguments, no name mangling, no post-processing.
  - `wp.ModuleBuildOptions(extra_cuda_include_dirs=..., extra_cuda_preamble=...)` injects
    `#include <optix_device.h>` and the OptiX include path into Warp's NVRTC run.
  - `@wp.func_native` snippets can wrap arbitrary device code, including calls to the
    OptiX intrinsics.
  - `wp.compile_aot_module(..., use_ptx=True)` returns the PTX path; the PTX is verbatim
    NVRTC output (virtual arch, C++17), self-contained except for the deliberately
    unresolved `_optix_*` symbols. Warp's own module loader must never be used for these
    modules (it would reject unresolved externs); the cubin path (nvPTXCompiler) must never
    be requested (same rejection as ptxas).
- **F4 - Launch semantics.** `optixLaunch` runs exactly width x height x depth threads, one
  per launch index, but the thread-to-launch-index mapping is implementation-defined
  (neighbor indices may land in different warps; 2D launches are tiled). Warp's `wp.tid()`
  (linear `blockIdx*blockDim+threadIdx`) is therefore NOT the launch index - and the
  `external_constant_params` ABI rejects `wp.tid()` by design. Kernels must use
  `optixGetLaunchIndex()` / `optixGetLaunchDimensions()`.
- **F5 - Constant-memory launch params.** Pipeline launch params live in a `__constant__`
  global named via `pipelineLaunchParamsVariableName`, 64 KB budget. The Warp ABI requires
  exactly one shared `@wp.struct` params type per module.
- **F6 - Precedent.** NVIDIA's 2022 numba extension for PyOptiX compiled Python functions to
  PTX and loaded them via OptiX successfully (it needed PTX entry-name string surgery; Warp
  needs none). The Warp feature was validated against NVIDIA's `otk-pyoptix` as an external
  consumer.
- **F7 - Licensing.** Hand-written/header-derived API declarations re-expressed in Python,
  with headers fetched at CI time from `NVIDIA/optix-dev` and never committed or shipped, are
  consistent with the precedents set by NVIDIA's own `cuda-python`, `nvidia-ml-py`, and
  `otk-pyoptix` (which ships `OptiX_LICENSE.txt` alongside permissive code). See section 9.

## 3. Architecture decisions

### D1 - Host bindings: pure ctypes with codegen'd declarations

- Load the driver library, call `optixQueryFunctionTable(ABI, ...)` and bind ~40 `CFUNCTYPE`
  entries; structs/enums as plain `ctypes.Structure` / `IntEnum`.
- Source of truth is a header-derived declaration dataset; at development time a generator
  emits plain `ctypes.Structure` classes from it, and the generated Python files are
  committed. No runtime data-driven marshaller.
- Per-version declaration data (structs, enums, constants AND function-table layout per ABI
  version) enables runtime ABI selection against `OPTIX_VERSION` - supporting OptiX 9.x
  without recompilation.
- Use `CDLL` semantics (ctypes releases the GIL for foreign calls; never `PyDLL`).
- Known landmines, each with a dedicated test: log-callback `CFUNCTYPE` must be pinned for
  the context lifetime and not fire after interpreter shutdown; function-table entry order
  and count must match exactly (mismatch = silent corruption); anonymous unions
  (`OptixBuildInput`) and struct churn across versions must be conformance-tested on Linux
  AND Windows (ctypes native alignment matches OptiX on GCC/MSVC, but that must be proven,
  not assumed).

### D2 - CUDA runtime: Warp is the core dependency

Warp (>= 1.17, pinned minor) provides the CUDA runtime layer: context/device management,
`wp.array` buffers (`.ptr` as `CUdeviceptr`), `wp.Stream` (its handle passed as `CUstream`
into the ctypes OptiX calls), and the NVRTC toolchain for kernel compilation. Since Warp is
load-bearing for kernel authoring (D3), splitting the runtime across two libraries is not
justified. Required dependencies: `warp-lang`, `numpy`. Duck-typed buffer support stays:
anything exposing `__cuda_array_interface__` / a raw `.ptr` (PyTorch, CuPy) is accepted.
CuPy's private `_NVRTCProgram` usage is deleted.

### D3 - Device code: Warp kernels as the primary authoring path

The library ships a device layer (working name `optix.device`) providing:

- Warp-callable wrappers for the OptiX intrinsics, implemented via `@wp.func_native`:
  `trace(...)`, `get_launch_index()`, `get_launch_dimensions()`, `get_sbt_data_pointer()`,
  `report_intersection(...)`, `get_triangle_barycentrics()`, payload get/set, and the rest
  of the device API surface.
- A program declaration API: decorate a Python function, get an OptiX program with correct
  entry name, params struct, payload type declarations, and SBT data binding.
- A compile bridge: warp module -> `wp.compile_aot_module(use_ptx=True)` -> `optixModuleCreate`
  with `pipelineLaunchParamsVariableName="params"`. The library owns the bridge so that the
  "never wp.launch / never cubin / never Warp's loader" rules (F3) are enforced internally
  and survive Warp API churn.
- Escape hatch unchanged: raw CUDA C++ source, PTX, or OptixIR handed straight to
  `optixModuleCreate` (replacing today's cupy-NVRTC path).

Note on `tid()` and barriers: under the `external_constant_params` ABI Warp rejects `wp.tid()`
and the tile API, and OptiX programs may not use shared memory at all (F4). The library will
add convenience functions via `@wp.func_native` where this is safe and semantically
well-defined - e.g. a launch-index-derived convenience accessor - but raw `wp.tid()` and
cross-thread primitives remain out of scope for OptiX programs. They remain fully available
in ordinary (non-OptiX) Warp kernels, which can coexist in the same application.

### D4 - Tiered host API

- **Tier 1 `optix.host`** - faithful, mostly 1:1 mirror of the OptiX C API over ctypes.
  Internal, but importable and stable; this is the foundation everything else uses.
- **Tier 2** - the refactored pythonic layer (the current public classes, rebuilt on tier 1):
  dataclass-style options objects with sane defaults; factory-first `ProgramGroup`
  (retiring the 16-keyword-arg `__init__`); automatic buffer/scratch management for
  acceleration-structure builds and the denoiser; the record system (SBT records /
  launch params) rebuilt on top of the Warp struct model instead of numpy structured arrays;
  context/stream plumbing through Warp.
- **Tier 3 (optional)** - high-level `Scene` / `Mesh` / `Camera` layer, ported and extended
  from the current `examples/sutil` code.

### D5 - Packaging and distribution

- Universal pure-Python wheel (`py3-none-any`), no build matrix. sdist for from-source
  installs is secondary.
- Acceptance criterion: `pip install python-optix` plus an NVIDIA driver plus a GPU is a
  working installation. No Cython, no host compiler, no CUDA Toolkit, no OptiX SDK download
  for the install itself.
- OptiX headers are fetched at runtime (with caching and `OPTIX_PATH` override) only when
  device code must be NVRTC-compiled and the user has not supplied include paths.
- README/build docs rewritten (current README is stale: claims OptiX 7.7.0 and an
  `OPTIX_EMBED_HEADERS` option that no longer exists in code).

### D6 - Testing and CI

- A pytest suite is a prerequisite, not an afterthought (the repo currently has none).
- **Conformance tests (CPU-only, blocking gate on free CI runners):** libclang-parses the
  downloaded OptiX headers and verifies every generated struct size/offset, enum value,
  constant, and the function-table layout for each supported ABI version. Runs on Linux and
  Windows.
- **GPU smoke tests (self-hosted runner, non-blocking):** the example set (triangle,
  spheres, dynamic geometry, denoiser, opacity micromaps) run end-to-end. GitHub-hosted GPU
  runners are paid-tier only; do not depend on them.
- Version gate: CI matrix over supported OptiX versions via the per-version declaration data.

### D7 - Release strategy

Breaking 2.0 release. Tier-2 class names stay stable where cheap; deprecation shims
elsewhere. The 1.x Cython line remains on `feature/optix_9_0_0_support`.

## 4. Target API sketch (illustrative, not normative)

Device side - programs authored in Python:

    import warp as wp
    import python_optix as ox

    class Params(wp.Struct):
        image: wp.array(dtype=wp.vec3)
        width: wp.int32
        height: wp.int32

    @wp.kernel(name="__raygen__rg", entry_point_abi="external_constant_params",
               module_options={"strip_hash": True}, enable_backward=False)
    def raygen():
        idx = ox.device.get_launch_index()
        ...ox.device.trace(...)

Host side - the existing tier-2 object model, now Warp-backed:

    ctx = ox.DeviceContext()
    module = ox.Module.from_programs(ctx, [raygen, miss, closest_hit])
    pipeline = ox.Pipeline(...); sbt = ox.ShaderBindingTable(...)
    pipeline.launch(sbt, dimensions=(width, height), params=params_array, stream=stream)

## 5. Phased roadmap

### Phase 0 - Feasibility spike (1-2 days, blocks everything)
Minimal end-to-end proof on real hardware: trivial Warp raygen (writes launch-index-derived
colors via SBT data) -> `wp.compile_aot_module(use_ptx=True)` -> ctypes
`optixQueryFunctionTable` + `optixModuleCreate` -> `optixLaunch` -> correct output image.
Risks being retired: `optix_device.h` compiling cleanly inside Warp's generated TU (its
`crt.h`/`half` vs `vector_types.h` interplay) and OptiX 9 driver acceptance of the PTX.
**Gate: spike succeeds, or D3 falls back to a small Python->CUDA-string emitter + raw NVRTC
(decision documented before any further work).**

### Phase 1 - ctypes host layer (parallelizable across modules)
Declaration dataset + codegen; dlopen/function-table engine; error handling; log callback;
per-version data. Port order: context -> build -> pipeline/program groups -> module ->
shader binding table -> denoiser -> opacity micromaps. Conformance tests land with each
module, not after.

### Phase 2 - Warp runtime + device layer
DeviceContext on Warp; `wp.array`/`wp.Stream` plumbing; NVRTC bridge via Warp; the raw
CUDA/PTX escape hatch; the `optix.device` intrinsic wrappers and program declaration API
(spike code promoted into the library).

### Phase 3 - Tier-2 refactor + example port
Rebuild the pythonic layer on tiers 1-2; port all examples (hello, triangle, spheres,
dynamic_geometry, dynamic_materials, opacity_micromap, denoiser, compile_with_tasks) with at
least the flagship subset (triangle, spheres, dynamic geometry) authored in Python kernels.
Examples are the acceptance suite.

### Phase 4 - Tier-3 high-level API
Scene/Mesh/Camera layer ported from `examples/sutil`.

### Phase 5 - Packaging, docs, CI
Universal wheel, README rewrite, provenance docs, conformance CI on free runners, GPU smoke
tests on self-hosted runner, PyPI release of 2.0.

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `optix_device.h` does not compile under Warp's NVRTC flags | Low (prototype existed) | Phase 0 spike; fallback emitter (D3) |
| Warp external-ABI is experimental and may change | High over time | Pin warp-lang minor; isolate all Warp ABI usage behind `optix.device`/compile bridge; conformance tests catch drift |
| Function-table/struct declaration errors corrupt memory silently | Medium | libclang conformance tests as blocking CI gate, both platforms |
| OptiX 9.2/10 changes ABI or launch-params rules (params size becomes mandatory next major) | Certain eventually | Per-version declaration data designed for this from day one |
| No GPU in CI for host-layer regressions | Certain | CPU-only conformance suite; self-hosted GPU smoke tests |
| Warp stream semantics with OptiX-launched work (issue #1618 class of bugs) | Medium | Library owns stream lifecycle for OptiX calls; documented sync discipline; smoke tests cover async paths |

## 7. Explicit non-goals (for 2.0)

- Autodiff through OptiX programs (Warp adjoint of ray tracing) - out of scope.
- Numba-based kernel authoring - superseded by the Warp path; revisit only if demand appears.
- CPU rendering fallback - OptiX requires a GPU and driver; Warp's CPU backend is irrelevant
  here.
- Contributing the low-level layer to NVIDIA's `otk-pyoptix` - it is sdist-only pybind11 with
  the same build pain this refactor eliminates; the differentiator here is the pure-Python
  wheel plus the Warp kernel path. Documented so the decision is revisit-able, not forgotten.

## 8. Open questions

1. Exact minimum Warp version to pin (1.17.0 assumed; re-evaluate at Phase 0).
2. Public naming of the device layer (`optix.device` vs `python_optix.wp`) - decide at
   Phase 2.
3. Whether OptiX 9.0 support is retained alongside 9.1 via the per-version data, or 2.0
   targets 9.1 only (9.1 makes the pipeline launch-params-size setting that becomes
   mandatory in the next major; leaning 9.1-only unless per-version support falls out cheap).
4. Windows support tier for 2.0 (conformance-tested per D6, but examples/GPU smoke may remain
   Linux-only initially).

## 9. License and provenance policy

- Never commit or ship OptiX header text, doc comments, or CI-generated declaration data.
- Headers are downloaded from `NVIDIA/optix-dev` at CI time (conformance) and at runtime
  (NVRTC device compilation, cached, `OPTIX_PATH` override respected).
- The repo documents that declarations are derived from OptiX X.Y headers and carries the
  applicable NVIDIA license text alongside the MIT package license, following the
  `otk-pyoptix` precedent (`OptiX_LICENSE.txt` shipped with permissive binding code).
- Commercial-use notification obligation (OptiX EULA) noted in the README.