using PassiveTracerFlows, Printf, Plots, JLD2
using Random: seed!

dev = CPU()
nothing # hide

      n = 128            # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 2.5e-3         # timestep
nothing # hide

L = 2π                   # domain size
μ = 5e-2                 # bottom drag
β = 5                    # the y-gradient of planetary PV

nlayers = 2              # number of layers
f₀, g = 1, 1             # Coriolis parameter and gravitational constant
 H = [0.2, 0.8]          # the rest depths of each layer
 ρ = [4.0, 5.0]          # the density of each layer

 U = zeros(nlayers) # the imposed mean zonal flow in each layer
 U[1] = 1.0
 U[2] = 0.0
nothing # hide

MQGprob = MultiLayerQG.Problem(nlayers, dev;
                               nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,
                               dt, stepper, aliased_fraction=0)
grid = MQGprob.grid
x, y = grid.x, grid.y

seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * ArrayType(dev)(randn((grid.nx, grid.ny, nlayers)))
q₀h = MQGprob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀  = irfft(q₀h, grid.nx, (1, 2))                    # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(MQGprob, q₀)
nothing # hide

κ = 0.002                        # Constant diffusivity
nsteps = 4000                    # total number of time-steps
tracer_release_time = 25.0       # run flow for some time before releasing tracer

ADprob = TracerAdvectionDiffusion.Problem(dev, MQGprob; κ, stepper, tracer_release_time)

gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))

amplitude, spread = 10, 0.15
c₀ = [amplitude * gaussian(x[i], y[j], spread) for j=1:grid.ny, i=1:grid.nx]

TracerAdvectionDiffusion.set_c!(ADprob, c₀)

sol, clock, vars, params, grid = ADprob.sol, ADprob.clock, ADprob.vars, ADprob.params, ADprob.grid
x, y = grid.x, grid.y

function get_concentration(prob)
  invtransform!(prob.vars.c, deepcopy(prob.sol), prob.params.MQGprob.params)

  return prob.vars.c
end

function get_streamfunction(prob)
  params, vars, grid = prob.params.MQGprob.params, prob.params.MQGprob.vars, prob.grid

  @. vars.qh = prob.params.MQGprob.sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  invtransform!(vars.ψ, vars.ψh, params)

  return vars.ψ
end

output = Output(ADprob, "advection-diffusion.jld2",
                (:concentration, get_concentration), (:streamfunction, get_streamfunction))

saveproblem(output)

save_frequency = 50 # frequency at which output is saved

startwalltime = time()
while clock.step <= nsteps
  if clock.step % save_frequency == 0
    saveoutput(output)
    log = @sprintf("Output saved, step: %04d, t: %.2f, walltime: %.2f min",
                   clock.step, clock.t, (time()-startwalltime) / 60)

    println(log)
  end

  stepforward!(ADprob)
  stepforward!(params.MQGprob)
  MultiLayerQG.updatevars!(params.MQGprob)
end

file = jldopen(output.path)

iterations = parse.(Int, keys(file["snapshots/t"]))
t = [file["snapshots/t/$i"] for i ∈ iterations]

layer = 2

c = [file["snapshots/concentration/$i"][:, :, layer] for i ∈ iterations]
ψ = [file["snapshots/streamfunction/$i"][:, :, layer] for i ∈ iterations]
nothing # hide

for i in 1:length(ψ)
  ψ[i] *= (amplitude / 5) / maximum(abs, ψ[i])
end

x,  y  = file["grid/x"],  file["grid/y"]
Lx, Ly = file["grid/Lx"], file["grid/Ly"]

plot_args = (xlabel = "x",
             ylabel = "y",
             aspectratio = 1,
             framestyle = :box,
             xlims = (-Lx/2, Lx/2),
             ylims = (-Ly/2, Ly/2),
             legend = :false,
             clims = (-amplitude/5, amplitude/5),
             colorbar_title = "\n concentration",
             color = :balance)

p = heatmap(x, y, Array(c[1]'), title = "concentration, t = " * @sprintf("%.2f", t[1]); plot_args...)
contour!(p, x, y, Array(ψ[1]'), levels = 0.15:0.3:1.5, lw=2, c=:grey, ls=:solid, alpha=0.5)
contour!(p, x, y, Array(ψ[1]'), levels = -0.15:-0.3:-1.5, lw=2, c=:grey, ls=:dash, alpha=0.5)

nothing # hide

anim = @animate for i ∈ 1:length(t)
  p[1][:title] = "concentration, t = " * @sprintf("%.2f", t[i])
  p[1][1][:z] = Array(c[i])
  p[1][2][:z] = Array(ψ[i])
  p[1][3][:z] = Array(ψ[i])
end

mp4(anim, "turbulentflow_advection-diffusion.mp4", fps = 12)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

