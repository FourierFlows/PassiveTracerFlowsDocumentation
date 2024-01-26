using PassiveTracerFlows, CairoMakie, Printf, JLD2, LinearAlgebra

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution = n²
stepper = "RK4"          # timestepper
     dt = 0.02           # timestep
 nsteps = 5000           # total number of time-steps
nothing # hide

L = 2π       # domain size
κ = 0.01     # diffusivity
nothing # hide

u(x) = 0.05
advecting_flow = OneDAdvectingFlow(; u, steadyflow = true)

prob = TracerAdvectionDiffusion.Problem(dev, advecting_flow; nx=n, Lx=L, κ, dt, stepper)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x = grid.x

gaussian(x, σ) = exp(-x^2 / 2σ^2)

amplitude, spread = 1, 0.15
c₀ = [amplitude * gaussian(x[i], spread) for i in 1:grid.nx]

TracerAdvectionDiffusion.set_c!(prob, c₀)
nothing #hide

function get_concentration(prob)
  ldiv!(prob.vars.c, prob.grid.rfftplan, deepcopy(prob.sol))

  return prob.vars.c
end

output = Output(prob, "advection-diffusion1D.jld2",
                (:concentration, get_concentration))

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

  stepforward!(prob)
end

file = jldopen(output.path)

iterations = parse.(Int, keys(file["snapshots/t"]))

t = [file["snapshots/t/$i"] for i ∈ iterations]
c = [file["snapshots/concentration/$i"] for i ∈ iterations]
nothing # hide

x, Lx = file["grid/x"], file["grid/Lx"]

n = Observable(1)
c_anim = @lift Array(c[$n])
title = @lift @sprintf("concentration, t = %s", t[$n])

fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "c",
          limits = ((-Lx/2, Lx/2), (0, maximum(c[1]))))

lines!(ax, x, c_anim; linewidth = 4)

frames = 1:length(t)
record(fig, "1D_advection-diffusion.mp4", frames, framerate = 18) do i
    n[] = i
end

nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
