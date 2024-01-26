using PassiveTracerFlows, Printf, CairoMakie, JLD2
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
f₀ = 1                   # Coriolis parameter
H = [0.2, 0.8]           # the rest depths of each layer
b = [-1.0, -1.2]         # Boussinesq buoyancy of each layer

U = zeros(nlayers)       # the imposed mean zonal flow in each layer
U[1] = 1.0
U[2] = 0.0
nothing # hide

MQGprob = MultiLayerQG.Problem(nlayers, dev;
                               nx=n, Lx=L, f₀, H, b, U, μ, β,
                               dt, stepper, aliased_fraction=0)

nx, ny = MQGprob.grid.nx, MQGprob.grid.ny

seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * device_array(dev)(randn((nx, ny, nlayers)))
q₀h = MQGprob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀  = irfft(q₀h, nx, (1, 2))                         # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(MQGprob, q₀)
nothing # hide

κ = 0.002                        # constant diffusivity
nsteps = 4000                    # total number of time-steps
tracer_release_time = 25.0       # run flow for some time before releasing tracer

ADprob = TracerAdvectionDiffusion.Problem(MQGprob; κ, stepper, tracer_release_time)

sol, clock, vars, params, grid = ADprob.sol, ADprob.clock, ADprob.vars, ADprob.params, ADprob.grid
x, y = grid.x, grid.y

gaussian(x, y, σ) = exp(-(x^2 + y^2) / 2σ^2)

amplitude, spread = 10, 0.15
c₀ = [amplitude * gaussian(x[i], y[j], spread) for j=1:grid.ny, i=1:grid.nx]

TracerAdvectionDiffusion.set_c!(ADprob, c₀)

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

for i in 1:lastindex(ψ)
  ψ[i] /= maximum(abs, ψ[i])
end

x,  y  = file["grid/x"],  file["grid/y"]
Lx, Ly = file["grid/Lx"], file["grid/Ly"]

n = Observable(1)

c_anim = @lift Array(c[$n])
ψ_anim = @lift Array(ψ[$n])
title = @lift @sprintf("concentration, t = %.2f", t[$n])

fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = title,
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

hm = heatmap!(ax, x, y, c_anim;
              colormap = :balance, colorrange = (-1, 1))
contour!(ax, x, y, ψ_anim;
         levels = 0.1:0.2:1, color = :grey, linestyle = :solid)
contour!(ax, x, y, ψ_anim;
         levels = -0.1:-0.2:-1, color = (:grey, 0.8), linestyle = :dash)

nothing # hide

frames = 1:length(t)
record(fig, "turbulentflow_advection-diffusion.mp4", frames, framerate = 12) do i
    n[] = i
end

nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
