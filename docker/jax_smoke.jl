# Build-time check of the Python/JAX layer (docker/Dockerfile). Fatal: Reactant traces a JAX
# function called through PythonCall. Reported only: a Pallas TPU kernel lowered by JAX is
# embedded into a Reactant module (the builder has no TPU, so nothing is compiled for it).
ENV["JAX_PLATFORMS"] = "cpu"   # JAX must not open a TPU that Reactant uses
using Reactant, PythonCall
Reactant.set_default_backend("cpu")

jax = pyimport("jax")
println("python ", pyimport("sys").version, ", jax ", jax.__version__, ", PythonCall ",
        pkgversion(PythonCall), ", Reactant ", pkgversion(Reactant))

s = @jit jax.numpy.sum(Reactant.to_rarray(Float32[1, 2, 3]))
s ≈ 6 || error("jax.numpy.sum through Reactant gave $s")
println("jax.numpy.sum through Reactant: ", s)

const PALLAS_TEXT = Ref("")
pallas_double(x) = Reactant.Ops.hlo_call(PALLAS_TEXT[], x)[1]

try
    pyexec("""
    import jax, jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(x_ref, o_ref):
        o_ref[...] = 2 * x_ref[...]

    def double(x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    try:
        text = jax.jit(double).trace(spec).lower(lowering_platforms=("tpu",)).as_text()
        how = "jit(...).trace(...).lower(lowering_platforms=('tpu',))"
    except Exception as e:
        print("trace/lower failed:", repr(e)[:500])
        text = jax.export.export(jax.jit(double), platforms=("tpu",))(spec).mlir_module()
        how = "jax.export.export(..., platforms=('tpu',))"
    """, Main)
    PALLAS_TEXT[] = pyeval(String, "text", Main)
    println("Pallas kernel lowered for TPU with ", pyeval(String, "how", Main))
    occursin("tpu_custom_call", PALLAS_TEXT[]) || error("no tpu_custom_call in the lowered module")
    mod = @code_hlo pallas_double(Reactant.to_rarray(zeros(Float32, 8, 128)))
    occursin("tpu_custom_call", string(mod)) || error("the Reactant module lost the tpu_custom_call")
    println("PALLAS CHECK ok: Pallas TPU kernel embedded in a Reactant module")
catch err
    msg = sprint(showerror, err)
    println("PALLAS CHECK FAILED (not fatal): ", first(msg, 3000))
end
