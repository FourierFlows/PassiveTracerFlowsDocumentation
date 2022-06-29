using PassiveTracerFlows, Plots, Printf

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution = n²
stepper = "RK4"          # timestepper
     dt = 0.02           # timestep
 nsteps = 800            # total number of time-steps
 nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π        # domain size
κ = 0.002     # diffusivity
nothing # hide

grid = TwoDGrid(n, L)

ψ₀ = 0.2
mx, my = 1, 1

ψ = [ψ₀ * cos(mx * grid.x[i]) * cos(my * grid.y[j]) for i in 1:grid.nx, j in 1:grid.ny]

uvel(x, y) =  ψ₀ * my * cos(mx * x) * sin(my * y)
vvel(x, y) = -ψ₀ * mx * sin(mx * x) * cos(my * y)
advecting_flow = TwoDAdvectingFlow(; u = uvel, v = vvel, steadyflow = true)
nothing # hide

prob = TracerAdvectionDiffusion.Problem(dev, advecting_flow; nx=n, Lx=L, κ, dt, stepper)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))

amplitude, spread = 0.5, 0.15
c₀ = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]

TracerAdvectionDiffusion.set_c!(prob, c₀)
nothing # hide

function plot_output(prob)
  c = prob.vars.c

  p = heatmap(x, y, Array(vars.c'),
         aspectratio = 1,
              c = :balance,
         legend = :false,
           clim = (-0.2, 0.2),
          xlims = (-grid.Lx/2, grid.Lx/2),
          ylims = (-grid.Ly/2, grid.Ly/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial tracer concentration (shading) + streamlines",
     framestyle = :box)

  contour!(p, x, y, Array(ψ'),
     levels=0.0125:0.025:0.2,
     lw=2, c=:black, ls=:solid, alpha=0.7)

  contour!(p, x, y, Array(ψ'),
     levels=-0.1875:0.025:-0.0125,
     lw=2, c=:black, ls=:dash, alpha=0.7)

  return p
end
nothing # hide

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

 if j % (200 / nsubs) == 0
    log = @sprintf("step: %04d, t: %d, walltime: %.2f min",
                   clock.step, clock.t, (time()-startwalltime)/60)

    println(log)
  end

  p[1][1][:z] = Array(vars.c)
  p[1][:title] = "concentration, t=" * @sprintf("%.2f", clock.t)

  stepforward!(prob, nsubs)
  TracerAdvectionDiffusion.updatevars!(prob)
end

mp4(anim, "cellularflow_advection-diffusion.mp4", fps=12)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

