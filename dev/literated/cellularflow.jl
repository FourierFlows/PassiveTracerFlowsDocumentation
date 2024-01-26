using PassiveTracerFlows, CairoMakie, Printf

dev = CPU()     # Device (CPU/GPU)
nothing # hide

     nx = 128            # 2D resolution = nx²
stepper = "RK4"          # timestepper
     dt = 0.02           # timestep
 nsteps = 800            # total number of time-steps
 nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

Lx = 2π       # domain size
κ = 0.002     # diffusivity
nothing # hide

grid = TwoDGrid(dev; nx, Lx)

ψ₀ = 0.2
mx, my = 1, 1

ψ = [ψ₀ * cos(mx * grid.x[i]) * cos(my * grid.y[j]) for i in 1:grid.nx, j in 1:grid.ny]

uvel(x, y) =  ψ₀ * my * cos(mx * x) * sin(my * y)
vvel(x, y) = -ψ₀ * mx * sin(mx * x) * cos(my * y)
advecting_flow = TwoDAdvectingFlow(; u = uvel, v = vvel, steadyflow = true)
nothing # hide

prob = TracerAdvectionDiffusion.Problem(dev, advecting_flow; nx, Lx, κ, dt, stepper)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))

amplitude, spread = 0.5, 0.15
c₀ = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]

TracerAdvectionDiffusion.set_c!(prob, c₀)
nothing # hide

c_anim = Observable(Array(vars.c))
title = Observable(@sprintf("concentration, t = %.2f", clock.t))

Lx, Ly = grid.Lx, grid.Ly

fig = Figure(size = (600, 600))

ax = Axis(fig[1, 1],
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = title,
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

hm = heatmap!(ax, x, y, c_anim;
              colormap = :balance, colorrange = (-0.2, 0.2))

contour!(ax, x, y, ψ;
         levels =  0.0125:0.025:0.2, color = :grey, linestyle = :solid)
contour!(ax, x, y, ψ;
         levels = -0.1875:0.025:-0.0125, color = (:grey, 0.8), linestyle = :dash)

fig

startwalltime = time()

frames = 0:round(Int, nsteps/nsubs)
record(fig, "cellularflow_advection-diffusion.mp4", frames, framerate = 12) do j
   if j % (200 / nsubs) == 0
      log = @sprintf("step: %04d, t: %d, walltime: %.2f min",
                     clock.step, clock.t, (time()-startwalltime)/60)

      println(log)
    end

  c_anim[] = vars.c
  title[] = @sprintf("concentration, t = %.2f", clock.t)

  stepforward!(prob, nsubs)
  TracerAdvectionDiffusion.updatevars!(prob)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
