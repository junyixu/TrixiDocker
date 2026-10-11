# TrixiDocker

GPU image for Runpod pods with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and CUDA.jl installed and precompiled,
so a pod skips the long `Pkg.instantiate` / precompile at boot.

```
ghcr.io/junyixu/trixidocker:latest
```

Built by [`.github/workflows/Docker.yml`](.github/workflows/Docker.yml) on every push to `main`
that touches `docker/`. A `feat/docker-<name>` branch publishes `:<name>` for testing.

## Environments

| Path | Selected by | Contents |
|------|-------------|----------|
| `/opt/trixi-env` | `JULIA_PROJECT` (default) | Trixi, Reactant, CUDA, KernelAbstractions, Enzyme, OrdinaryDiffEqLowStorageRK, OrdinaryDiffEqSSPRK, StaticArrays, BenchmarkTools, Trixi2Vtk, CairoMakie |
| `/opt/trixicuda-env` | `julia --project=$TRIXICUDA_PROJECT` | TrixiCUDA.jl (pins Trixi.jl ≤ 0.13.5), CUDA, OrdinaryDiffEq solvers |

Preferences baked in (`docker/*/LocalPreferences.toml`; changing them triggers recompilation):

- main: Trixi `backend = "kernelabstractions"`, `loop_vectorization = false`
  (the configuration Enzyme and Reactant can trace); Reactant_jll CUDA 12.9 build;
  CUDA toolkit 12.9.
- trixicuda: the `log`/`sqrt`/`loop_vectorization` values TrixiCUDA.jl sets on load.

## Pod requirements

The CUDA 12.9 binaries need a host driver ≥ 575. When creating the pod, restrict the host
CUDA version accordingly (e.g. `allowedCudaVersions = ["12.9", "13.0", "13.1"]`).

## Updating packages

Edit `docker/<env>/Project.toml`, re-resolve `Manifest.toml` with the Julia version from the
`Dockerfile` (`JULIA_VERSION`), and push.
