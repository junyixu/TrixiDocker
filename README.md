# TrixiDocker

GPU image for Runpod pods with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and CUDA.jl installed and precompiled,
so a pod skips the long `Pkg.instantiate` / precompile at boot.

| Tag | For | Reactant_jll build |
|-----|-----|--------------------|
| `ghcr.io/junyixu/trixidocker:latest` | NVIDIA GPU pods (Runpod) | CUDA 12.9 |
| `ghcr.io/junyixu/trixidocker:tpu` | Cloud TPU VMs | CPU build, drives the TPU through libtpu (pre-downloaded) |

Built by [`.github/workflows/Docker.yml`](.github/workflows/Docker.yml) (one job per variant) on
every push to `main` that touches `docker/`. A `feat/docker-<name>` branch publishes `:<name>`
and `:<name>-tpu` for testing.

## Cloud TPU VM

```sh
sudo docker run --rm -it --privileged --net=host ghcr.io/junyixu/trixidocker:tpu julia
```

`--privileged --net=host` give the container the TPU devices (`/dev/vfio`). Inside the container
the host's `LD_PRELOAD` of tcmalloc (set in `/etc/environment` on TPU VM images), which crashes
Julia, does not apply. The `:tpu` image has no TrixiCUDA environment.

## Environments

| Path | Selected by | Contents |
|------|-------------|----------|
| `/opt/trixi-env` | `JULIA_PROJECT` (default) | Trixi, Reactant, CUDA, KernelAbstractions, Enzyme, OrdinaryDiffEqLowStorageRK, OrdinaryDiffEqSSPRK, StaticArrays, BenchmarkTools, Trixi2Vtk, CairoMakie |
| `/opt/trixicuda-env` | `julia --project=$TRIXICUDA_PROJECT` | TrixiCUDA.jl (pins Trixi.jl ≤ 0.13.5), CUDA, OrdinaryDiffEq solvers — `:latest` only |

Preferences baked in (`docker/*/LocalPreferences.toml`; changing them triggers recompilation):

- main: Trixi `backend = "kernelabstractions"`, `loop_vectorization = false`
  (the configuration Enzyme and Reactant can trace); Reactant_jll CUDA 12.9 build
  (`:tpu`: CPU build); CUDA toolkit 12.9.
- trixicuda: the `log`/`sqrt`/`loop_vectorization` values TrixiCUDA.jl sets on load.

## Pod requirements

The CUDA 12.9 binaries need a host driver ≥ 575. When creating the pod, restrict the host
CUDA version accordingly (e.g. `allowedCudaVersions = ["12.9", "13.0", "13.1"]`).

## Updating packages

Edit `docker/<env>/Project.toml`, re-resolve `Manifest.toml` with the Julia version from the
`Dockerfile` (`JULIA_VERSION`), and push.
