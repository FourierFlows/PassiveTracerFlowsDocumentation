var documenterSearchIndex = {"docs":
[{"location":"modules/traceradvectiondiffusion/#TracerAdvectionDiffusion-Module","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/#Basic-Equations","page":"TracerAdvectionDiffusion Module","title":"Basic Equations","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"This module solves the advection diffusion equation for a passive tracer concentration  c(x y t) in two-dimensions by an advecting flow bmu(x y t):","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t c + bmu bmcdot bmnabla c = underbraceeta partial_x^2 c + kappa partial_y^2 c_textrmdiffusivity + underbracekappa_h (-1)^n_h nabla^2n_hc_textrmhyper-diffusivity ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmu = (u v) is the two-dimensional advecting flow, eta the x-diffusivity and kappa is the y-diffusivity. If eta is not defined then the code uses isotropic diffusivity, i.e., eta partial_x^2 c + kappa partial_y^2 c mapsto kappa nabla^2. The advecting flow could be either compressible or incompressible. ","category":"page"},{"location":"modules/traceradvectiondiffusion/#Implementation","page":"TracerAdvectionDiffusion Module","title":"Implementation","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"The equation is time-stepped forward in Fourier space:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t widehatc = - widehatbmu bmcdot bmnabla c - left (eta k_x^2 + kappa k_y^2) + kappa_h bmk^2nu_h right widehatc ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmk = (k_x k_y).","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"Thus:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"beginaligned\nL  = -eta k_x^2 - kappa k_y^2 - kappa_h bmk^2nu_h  \nN(widehatc) = - mathrmFFT(u partial_x c + v partial_y c) \nendaligned","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"EditURL = \"https://github.com/FourierFlows/PassiveTracerFlowsDocumentation/blob/master/examples/cellularflow.jl\"","category":"page"},{"location":"generated/cellularflow/#Advection-diffusion-of-tracer-by-cellular-flow","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"This example can be viewed as a Jupyter notebook via (Image: ).","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"An example demonstrating the advection-diffusion of a tracer by a cellular flow.","category":"page"},{"location":"generated/cellularflow/#Install-dependencies","page":"Advection-diffusion of tracer by cellular flow","title":"Install dependencies","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"First let's make sure we have all required packages installed.","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"using Pkg\npkg\"add PassiveTracerFlows, Plots, Printf\"","category":"page"},{"location":"generated/cellularflow/#Let's-begin","page":"Advection-diffusion of tracer by cellular flow","title":"Let's begin","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Let's load PassiveTracerFlows.jl and some other needed packages.","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"using PassiveTracerFlows, Plots, Printf","category":"page"},{"location":"generated/cellularflow/#Choosing-a-device:-CPU-or-GPU","page":"Advection-diffusion of tracer by cellular flow","title":"Choosing a device: CPU or GPU","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"dev = CPU()     # Device (CPU/GPU)\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Numerical-parameters-and-time-stepping-parameters","page":"Advection-diffusion of tracer by cellular flow","title":"Numerical parameters and time-stepping parameters","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"      n = 128            # 2D resolution = n²\nstepper = \"RK4\"          # timestepper\n     dt = 0.02           # timestep\n nsteps = 800            # total number of time-steps\n nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Numerical-parameters-and-time-stepping-parameters-2","page":"Advection-diffusion of tracer by cellular flow","title":"Numerical parameters and time-stepping parameters","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"L = 2π        # domain size\nκ = 0.002     # diffusivity\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Set-up-cellular-flow","page":"Advection-diffusion of tracer by cellular flow","title":"Set up cellular flow","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We create a two-dimensional grid to construct the cellular flow. Our cellular flow is derived from a streamfunction ψ(x y) = ψ₀ cos(x) cos(y) as (u v) = (-_y ψ _x ψ).","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"grid = TwoDGrid(n, L)\nx, y = gridpoints(grid)\n\nψ₀ = 0.2\nmx, my = 1, 1\n\nψ = @. ψ₀ * cos(mx * x) * cos(my * y)\n\nuvel(x, y) =  ψ₀ * mx * cos(mx * x) * sin(my * y)\nvvel(x, y) = -ψ₀ * my * sin(mx * x) * cos(my * y)\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Problem-setup","page":"Advection-diffusion of tracer by cellular flow","title":"Problem setup","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We initialize a Problem by providing a set of keyword arguments.","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"prob = TracerAdvectionDiffusion.Problem(; nx=n, Lx=L, kap=κ, steadyflow=true, u=uvel, v=vvel,\n                                          dt=dt, stepper=stepper, dev=dev)\nnothing # hide","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"and define some shortcuts","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid\nx, y = grid.x, grid.y\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Setting-initial-conditions","page":"Advection-diffusion of tracer by cellular flow","title":"Setting initial conditions","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Our initial condition for the tracer c is a gaussian centered at (x y) = (L_x5 0).","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))\n\namplitude, spread = 0.5, 0.15\nc₀ = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]\n\nTracerAdvectionDiffusion.set_c!(prob, c₀)\nnothing # hide","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Let's plot the initial tracer concentration and streamlines. Note that when plotting, we decorate the variable to be plotted with Array() to make sure it is brought back on the CPU when vars live on the GPU.","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"function plot_output(prob)\n  c = prob.vars.c\n\n  p = heatmap(x, y, Array(vars.c'),\n         aspectratio = 1,\n              c = :balance,\n         legend = :false,\n           clim = (-0.2, 0.2),\n          xlims = (-grid.Lx/2, grid.Lx/2),\n          ylims = (-grid.Ly/2, grid.Ly/2),\n         xticks = -3:3,\n         yticks = -3:3,\n         xlabel = \"x\",\n         ylabel = \"y\",\n          title = \"initial tracer concentration (shading) + streamlines\",\n     framestyle = :box)\n\n  contour!(p, x, y, Array(ψ'),\n     levels=0.0125:0.025:0.2,\n     lw=2, c=:black, ls=:solid, alpha=0.7)\n\n  contour!(p, x, y, Array(ψ'),\n     levels=-0.1875:0.025:-0.0125,\n     lw=2, c=:black, ls=:dash, alpha=0.7)\n\n  return p\nend\nnothing # hide","category":"page"},{"location":"generated/cellularflow/#Time-stepping-the-Problem-forward","page":"Advection-diffusion of tracer by cellular flow","title":"Time-stepping the Problem forward","text":"","category":"section"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We time-step the Problem forward in time.","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"startwalltime = time()\n\np = plot_output(prob)\n\nanim = @animate for j = 0:round(Int, nsteps/nsubs)\n\n if j % (200 / nsubs) == 0\n    log = @sprintf(\"step: %04d, t: %d, walltime: %.2f min\",\n                   clock.step, clock.t, (time()-startwalltime)/60)\n\n    println(log)\n  end\n\n  p[1][1][:z] = Array(vars.c)\n  p[1][:title] = \"concentration, t=\" * @sprintf(\"%.2f\", clock.t)\n\n  stepforward!(prob, nsubs)\n  TracerAdvectionDiffusion.updatevars!(prob)\nend\n\nmp4(anim, \"cellularflow.mp4\", fps=12)","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"","category":"page"},{"location":"generated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#PassiveTracerFlows.jl-Documentation","page":"Home","title":"PassiveTracerFlows.jl Documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows.jl is a collection of modules which leverage the  FourierFlows.jl framework to solve for advection-diffusion problems on periodic domains.","category":"page"},{"location":"","page":"Home","title":"Home","text":"info: Unicode\nOftentimes unicode symbols are used in modules for certain variables or parameters. For  example, κ is commonly used to denote the diffusivity, or ∂ is used  to denote partial differentiation. Unicode symbols can be entered in the Julia REPL by  typing, e.g., \\kappa or \\partial followed by the tab key.Read more about Unicode symbols in the  Julia Documentation.","category":"page"},{"location":"#Developers","page":"Home","title":"Developers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows is currently being developed by Navid C. Constantinou and Gregory L. Wagner.","category":"page"},{"location":"#Cite","page":"Home","title":"Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The code is citable via zenodo, doi:10.5281/zenodo.2535983.","category":"page"},{"location":"man/types/#Private-types","page":"Private types","title":"Private types","text":"","category":"section"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:-2","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","text":"ConstDiffParams(eta, kap, kaph, nkaph, u, v)\nConstDiffParams(eta, kap, u, v)\n\nReturns the params for constant diffusivity problem with time-varying flow.\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","text":"ConstDiffSteadyFlowParams(eta, kap, kaph, nkaph, u, v, g)\nConstDiffSteadyFlowParams(eta, kap, u, v, g)\n\nReturns the params for constant diffusivity problem with time-steady flow.\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.Vars-Union{Tuple{T}, Tuple{Dev}, Tuple{Dev,AbstractGrid{T,A} where A}} where T where Dev","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.Vars","text":"Vars(g)\n\nReturns the vars for constant diffusivity problem on grid g.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"man/functions/#Functions-exported-from-PassiveTracerFlows:","page":"Functions","title":"Functions exported from PassiveTracerFlows:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#Functions-exported-from-TracerAdvectionDiffusion:","page":"Functions","title":"Functions exported from TracerAdvectionDiffusion:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.set_c!-Tuple{Any,Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.set_c!","text":"set_c!(prob, c)\n\nSet the solution sol as the transform of c and update variables v on the grid g.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!-Tuple{Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!","text":"updatevars!(prob)\n\nUpdate the vars in v on the grid g with the solution in sol.\n\n\n\n\n\n","category":"method"}]
}
