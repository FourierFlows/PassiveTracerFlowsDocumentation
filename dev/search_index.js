var documenterSearchIndex = {"docs":
[{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"EditURL = \"https://github.com/FourierFlows/PassiveTracerFlowsDocumentation/blob/main/examples/turbulent_advection-diffusion.jl\"","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Advection-diffusion-of-tracer-by-a-turbulent-flow","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"This is an example demonstrating the advection-diffusion of a tracer using a turbulent flow generated by the GeophysicalFlows.jl package.","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Install-dependencies","page":"Advection-diffusion of tracer by a turbulent flow","title":"Install dependencies","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"First let's make sure we have all the required packages installed","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"```julia using Pkg pkg.add([\"PassiveTracerFlows\", \"Printf\", \"Plots\", \"JLD2\"])","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Let's-begin","page":"Advection-diffusion of tracer by a turbulent flow","title":"Let's begin","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"First load PassiveTracerFlows.jl and the other packages needed to run this example.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"using PassiveTracerFlows, Printf, Plots, JLD2\nusing Random: seed!","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Choosing-a-device:-CPU-or-GPU","page":"Advection-diffusion of tracer by a turbulent flow","title":"Choosing a device: CPU or GPU","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"dev = CPU()\nnothing # hide","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Setting-up-a-MultiLayerQG.Problem-to-generate-a-turbulent-flow","page":"Advection-diffusion of tracer by a turbulent flow","title":"Setting up a MultiLayerQG.Problem to generate a turbulent flow","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"The tubulent flow we use to advect the passive tracer is generated using the MultiLayerQG module from the GeophysicalFlows.jl package. A more detailed setup of this two layer system is found at the GeophysicalFlows Documentation.","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Numerical-and-time-stepping-parameters-for-the-flow","page":"Advection-diffusion of tracer by a turbulent flow","title":"Numerical and time stepping parameters for the flow","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"      n = 128            # 2D resolution = n²\nstepper = \"FilteredRK4\"  # timestepper\n     dt = 2.5e-3         # timestep","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Physical parameters","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"L = 2π                   # domain size\nμ = 5e-2                 # bottom drag\nβ = 5                    # the y-gradient of planetary PV\n\nnlayers = 2              # number of layers\nf₀, g = 1, 1             # Coriolis parameter and gravitational constant\n H = [0.2, 0.8]          # the rest depths of each layer\n ρ = [4.0, 5.0]          # the density of each layer\n\n U = zeros(nlayers) # the imposed mean zonal flow in each layer\n U[1] = 1.0\n U[2] = 0.0","category":"page"},{"location":"literated/turbulent_advection-diffusion/#MultiLayerQG.Problem-setup,-shortcuts-and-initial-conditions","page":"Advection-diffusion of tracer by a turbulent flow","title":"MultiLayerQG.Problem setup, shortcuts and initial conditions","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"MQGprob = MultiLayerQG.Problem(nlayers, dev;\n                               nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,\n                               dt, stepper, aliased_fraction=0)\ngrid = MQGprob.grid\nx, y = grid.x, grid.y","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Initial conditions","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"seed!(1234) # reset of the random number generator for reproducibility\nq₀  = 1e-2 * ArrayType(dev)(randn((grid.nx, grid.ny, nlayers)))\nq₀h = MQGprob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2\nq₀  = irfft(q₀h, grid.nx, (1, 2))                    # apply irfft only in dims=1, 2\n\nMultiLayerQG.set_q!(MQGprob, q₀)\nnothing","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Tracer-advection-diffusion-setup","page":"Advection-diffusion of tracer by a turbulent flow","title":"Tracer advection-diffusion setup","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Now that we have a MultiLayerQG.Problem setup to generate our turbulent flow, we setup an advection-diffusion simulation. This is done by passing the MultiLayerQG.Problem as an argument to TracerAdvectionDiffusion.Problem which sets up an advection-diffusion problem with same parameters where applicable. We also need to pass a value for the constant diffusivity κ, the stepper used to step the problem forward and when we want the tracer released into the flow. We will let the flow run up to t = tracer_release_time and then release the tracer and let it evolve with the flow.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"κ = 0.002                        # Constant diffusivity\nnsteps = 4000                    # total number of time-steps\ntracer_release_time = 25.0       # run flow for some time before releasing tracer\n\nADprob = TracerAdvectionDiffusion.Problem(dev, MQGprob; κ, stepper, tracer_release_time)","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Initial-condition-for-concentration-in-both-layers","page":"Advection-diffusion of tracer by a turbulent flow","title":"Initial condition for concentration in both layers","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"We have a two layer system so we will advect-diffuse the tracer in both layers. To do this we set the initial condition for tracer concetration as a Gaussian centered at the origin. Then we create some shortcuts for the TracerAdvectionDiffusion.Problem.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))\n\namplitude, spread = 10, 0.15\nc₀ = [amplitude * gaussian(x[i], y[j], spread) for j=1:grid.ny, i=1:grid.nx]\n\nTracerAdvectionDiffusion.set_c!(ADprob, c₀)","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Shortcuts for advection-diffusion problem","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"sol, clock, vars, params, grid = ADprob.sol, ADprob.clock, ADprob.vars, ADprob.params, ADprob.grid\nx, y = grid.x, grid.y","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Saving-output","page":"Advection-diffusion of tracer by a turbulent flow","title":"Saving output","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"The parent package FourierFlows.jl provides the functionality to save the output from our simulation. To do this we write a function get_concentration and pass this to the Output function along with the TracerAdvectionDiffusion.Problem and the name of the output file.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"function get_concentration(prob)\n  invtransform!(prob.vars.c, deepcopy(prob.sol), prob.params.MQGprob.params)\n\n  return prob.vars.c\nend\n\nfunction get_streamfunction(prob)\n  params, vars, grid = prob.params.MQGprob.params, prob.params.MQGprob.vars, prob.grid\n\n  @. vars.qh = prob.params.MQGprob.sol\n\n  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)\n\n  invtransform!(vars.ψ, vars.ψh, params)\n\n  return vars.ψ\nend\n\noutput = Output(ADprob, \"advection-diffusion.jld2\",\n                (:concentration, get_concentration), (:streamfunction, get_streamfunction))","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"This saves information that we will use for plotting later on","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"saveproblem(output)","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Step-the-problem-forward-and-save-the-output","page":"Advection-diffusion of tracer by a turbulent flow","title":"Step the problem forward and save the output","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"We specify that we would like to save the concentration every save_frequency timesteps; then we step the problem forward.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"save_frequency = 50 # Frequency at which output is saved\n\nstartwalltime = time()\nwhile clock.step <= nsteps\n  if clock.step % save_frequency == 0\n    saveoutput(output)\n    log = @sprintf(\"Output saved, step: %04d, t: %.2f, walltime: %.2f min\",\n                   clock.step, clock.t, (time()-startwalltime) / 60)\n\n    println(log)\n  end\n\n  stepforward!(ADprob)\n  stepforward!(params.MQGprob)\n  MultiLayerQG.updatevars!(params.MQGprob)\nend","category":"page"},{"location":"literated/turbulent_advection-diffusion/#Visualising-the-output","page":"Advection-diffusion of tracer by a turbulent flow","title":"Visualising the output","text":"","category":"section"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"We now have output from our simulation saved in advection-diffusion.jld2. As a demonstration, we load the JLD2 output and create a time series for the tracer that has been advected-diffused in the lower layer of our fluid.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Create time series for the concentration in the upper layer","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"file = jldopen(output.path)\n\niterations = parse.(Int, keys(file[\"snapshots/t\"]))\nt = [file[\"snapshots/t/$i\"] for i ∈ iterations]","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Concentration and streamfunction time series in the bottom layer, layer = 2.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"layer = 2\n\nc = [file[\"snapshots/concentration/$i\"][:, :, layer] for i ∈ iterations]\nψ = [file[\"snapshots/streamfunction/$i\"][:, :, layer] for i ∈ iterations]","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"We normalize all streamfunctions to have maximum absolute value amplitude / 5.","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"for i in 1:length(ψ)\n  ψ[i] *= (amplitude / 5) / maximum(abs, ψ[i])\nend\n\nx,  y  = file[\"grid/x\"],  file[\"grid/y\"]\nLx, Ly = file[\"grid/Lx\"], file[\"grid/Ly\"]\n\nplot_args = (xlabel = \"x\",\n             ylabel = \"y\",\n             aspectratio = 1,\n             framestyle = :box,\n             xlims = (-Lx/2, Lx/2),\n             ylims = (-Ly/2, Ly/2),\n             legend = :false,\n             clims = (-amplitude/5, amplitude/5),\n             colorbar_title = \"\\n concentration\",\n             color = :balance)\n\np = heatmap(x, y, Array(c[1]'), title = \"concentration, t = \" * @sprintf(\"%.2f\", t[1]); plot_args...)\ncontour!(p, x, y, Array(ψ[1]'), levels = 0.15:0.3:1.5, lw=2, c=:grey, ls=:solid, alpha=0.5)\ncontour!(p, x, y, Array(ψ[1]'), levels = -0.15:-0.3:-1.5, lw=2, c=:grey, ls=:dash, alpha=0.5)\n\nnothing # hide","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"Create a movie of the tracer","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"anim = @animate for i ∈ 1:length(t)\n  p[1][:title] = \"concentration, t = \" * @sprintf(\"%.2f\", t[i])\n  p[1][1][:z] = Array(c[i])\n  p[1][2][:z] = Array(ψ[i])\n  p[1][3][:z] = Array(ψ[i])\nend\n\nmp4(anim, \"turbulentflow_advection-diffusion.mp4\", fps = 12)","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"","category":"page"},{"location":"literated/turbulent_advection-diffusion/","page":"Advection-diffusion of tracer by a turbulent flow","title":"Advection-diffusion of tracer by a turbulent flow","text":"This page was generated using Literate.jl.","category":"page"},{"location":"modules/traceradvectiondiffusion/#TracerAdvectionDiffusion-Module","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/#Basic-Equations","page":"TracerAdvectionDiffusion Module","title":"Basic Equations","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"This module solves the advection diffusion equation for a passive tracer concentration  c(x y t) in two-dimensions by an advecting flow bmu(x y t):","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t c + bmu bmcdot bmnabla c = underbraceeta partial_x^2 c + kappa partial_y^2 c_textrmdiffusivity + underbracekappa_h (-1)^n_h nabla^2n_hc_textrmhyper-diffusivity ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmu = (u v) is the two-dimensional advecting flow, eta the x-diffusivity and kappa is the y-diffusivity. If eta is not defined then the code uses isotropic diffusivity, i.e., eta partial_x^2 c + kappa partial_y^2 c mapsto kappa nabla^2. The advecting flow could be either compressible or incompressible. ","category":"page"},{"location":"modules/traceradvectiondiffusion/#Implementation","page":"TracerAdvectionDiffusion Module","title":"Implementation","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"The equation is time-stepped forward in Fourier space:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t widehatc = - widehatbmu bmcdot bmnabla c - left (eta k_x^2 + kappa k_y^2) + kappa_h bmk^2nu_h right widehatc ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmk = (k_x k_y).","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"Thus:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"beginaligned\nL  = -eta k_x^2 - kappa k_y^2 - kappa_h bmk^2nu_h  \nN(widehatc) = - mathrmFFT(u partial_x c + v partial_y c) \nendaligned","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"EditURL = \"https://github.com/FourierFlows/PassiveTracerFlowsDocumentation/blob/main/examples/cellularflow.jl\"","category":"page"},{"location":"literated/cellularflow/#Advection-diffusion-of-tracer-by-cellular-flow","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"This example can be viewed as a Jupyter notebook via (Image: ).","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"An example demonstrating the advection-diffusion of a tracer by a cellular flow.","category":"page"},{"location":"literated/cellularflow/#Install-dependencies","page":"Advection-diffusion of tracer by cellular flow","title":"Install dependencies","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"First let's make sure we have all required packages installed.","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"using Pkg\npkg\"add PassiveTracerFlows, Plots, Printf\"","category":"page"},{"location":"literated/cellularflow/#Let's-begin","page":"Advection-diffusion of tracer by cellular flow","title":"Let's begin","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Let's load PassiveTracerFlows.jl and some other needed packages.","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"using PassiveTracerFlows, Plots, Printf","category":"page"},{"location":"literated/cellularflow/#Choosing-a-device:-CPU-or-GPU","page":"Advection-diffusion of tracer by cellular flow","title":"Choosing a device: CPU or GPU","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"dev = CPU()     # Device (CPU/GPU)\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Numerical-parameters-and-time-stepping-parameters","page":"Advection-diffusion of tracer by cellular flow","title":"Numerical parameters and time-stepping parameters","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"      n = 128            # 2D resolution = n²\nstepper = \"RK4\"          # timestepper\n     dt = 0.02           # timestep\n nsteps = 800            # total number of time-steps\n nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Numerical-parameters-and-time-stepping-parameters-2","page":"Advection-diffusion of tracer by cellular flow","title":"Numerical parameters and time-stepping parameters","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"L = 2π        # domain size\nκ = 0.002     # diffusivity\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Set-up-cellular-flow","page":"Advection-diffusion of tracer by cellular flow","title":"Set up cellular flow","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We create a two-dimensional grid to construct the cellular flow. Our cellular flow is derived from a streamfunction ψ(x y) = ψ₀ cos(x) cos(y) as (u v) = (-_y ψ _x ψ).","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"grid = TwoDGrid(n, L)\n\nψ₀ = 0.2\nmx, my = 1, 1\n\nψ = [ψ₀ * cos(mx * grid.x[i]) * cos(my * grid.y[j]) for i in 1:grid.nx, j in 1:grid.ny]\n\nuvel(x, y) =  ψ₀ * my * cos(mx * x) * sin(my * y)\nvvel(x, y) = -ψ₀ * mx * sin(mx * x) * cos(my * y)\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Problem-setup","page":"Advection-diffusion of tracer by cellular flow","title":"Problem setup","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We initialize a Problem by providing a set of keyword arguments.","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"prob = TracerAdvectionDiffusion.Problem(dev; nx=n, Lx=L, κ=κ, steadyflow=true, u=uvel, v=vvel,\n                                          dt=dt, stepper=stepper)\nnothing # hide","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"and define some shortcuts","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid\nx, y = grid.x, grid.y\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Setting-initial-conditions","page":"Advection-diffusion of tracer by cellular flow","title":"Setting initial conditions","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Our initial condition for the tracer c is a gaussian centered at (x y) = (L_x5 0).","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))\n\namplitude, spread = 0.5, 0.15\nc₀ = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]\n\nTracerAdvectionDiffusion.set_c!(prob, c₀)\nnothing # hide","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"Let's plot the initial tracer concentration and streamlines. Note that when plotting, we decorate the variable to be plotted with Array() to make sure it is brought back on the CPU when vars live on the GPU.","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"function plot_output(prob)\n  c = prob.vars.c\n\n  p = heatmap(x, y, Array(vars.c'),\n         aspectratio = 1,\n              c = :balance,\n         legend = :false,\n           clim = (-0.2, 0.2),\n          xlims = (-grid.Lx/2, grid.Lx/2),\n          ylims = (-grid.Ly/2, grid.Ly/2),\n         xticks = -3:3,\n         yticks = -3:3,\n         xlabel = \"x\",\n         ylabel = \"y\",\n          title = \"initial tracer concentration (shading) + streamlines\",\n     framestyle = :box)\n\n  contour!(p, x, y, Array(ψ'),\n     levels=0.0125:0.025:0.2,\n     lw=2, c=:black, ls=:solid, alpha=0.7)\n\n  contour!(p, x, y, Array(ψ'),\n     levels=-0.1875:0.025:-0.0125,\n     lw=2, c=:black, ls=:dash, alpha=0.7)\n\n  return p\nend\nnothing # hide","category":"page"},{"location":"literated/cellularflow/#Time-stepping-the-Problem-forward","page":"Advection-diffusion of tracer by cellular flow","title":"Time-stepping the Problem forward","text":"","category":"section"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"We time-step the Problem forward in time.","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"startwalltime = time()\n\np = plot_output(prob)\n\nanim = @animate for j = 0:round(Int, nsteps/nsubs)\n\n if j % (200 / nsubs) == 0\n    log = @sprintf(\"step: %04d, t: %d, walltime: %.2f min\",\n                   clock.step, clock.t, (time()-startwalltime)/60)\n\n    println(log)\n  end\n\n  p[1][1][:z] = Array(vars.c)\n  p[1][:title] = \"concentration, t=\" * @sprintf(\"%.2f\", clock.t)\n\n  stepforward!(prob, nsubs)\n  TracerAdvectionDiffusion.updatevars!(prob)\nend\n\nmp4(anim, \"cellularflow_advection-diffusion.mp4\", fps=12)","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"","category":"page"},{"location":"literated/cellularflow/","page":"Advection-diffusion of tracer by cellular flow","title":"Advection-diffusion of tracer by cellular flow","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#PassiveTracerFlows.jl-Documentation","page":"Home","title":"PassiveTracerFlows.jl Documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows.jl is a collection of modules which leverage the  FourierFlows.jl framework to solve for advection-diffusion problems on periodic domains.","category":"page"},{"location":"","page":"Home","title":"Home","text":"info: Unicode\nOftentimes unicode symbols are used in modules for certain variables or parameters. For  example, κ is commonly used to denote the diffusivity, or ∂ is used  to denote partial differentiation. Unicode symbols can be entered in the Julia REPL by  typing, e.g., \\kappa or \\partial followed by the tab key.Read more about Unicode symbols in the  Julia Documentation.","category":"page"},{"location":"#Developers","page":"Home","title":"Developers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows is currently being developed by Navid C. Constantinou and Gregory L. Wagner.","category":"page"},{"location":"#Cite","page":"Home","title":"Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The code is citable via zenodo, doi:10.5281/zenodo.2535983.","category":"page"},{"location":"man/types/#Private-types","page":"Private types","title":"Private types","text":"","category":"section"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:-2","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","text":"struct ConstDiffParams{T} <: AbstractConstDiffParams\n\nA struct containing the parameters for a constant diffusivity problem with time-varying flow. Included are:\n\nη::Any\nisotropic horizontal diffusivity coefficient\nκ::Any\nisotropic vertical diffusivity coefficient\nκh::Any\nisotropic hyperdiffusivity coefficient\nnκh::Int64\nisotropic hyperdiffusivity order\nu::Function\nfunction returning the x-component of advecting flow\nv::Function\nfunction returning the y-component of advecting flow\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams-NTuple{4, Any}","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","text":"ConstDiffParams(η, κ, u, v)\n\nThe constructor for the params struct for constant diffusivity problem and time-varying flow.\n\n\n\n\n\n","category":"method"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","text":"struct ConstDiffParams{T} <: AbstractConstDiffParams\n\nA struct containing the parameters for a constant diffusivity problem with steady flow. Included are:\n\nη::Any\nisotropic horizontal diffusivity coefficient\nκ::Any\nisotropic vertical diffusivity coefficient\nκh::Any\nisotropic hyperdiffusivity coefficient\nnκh::Int64\nisotropic hyperdiffusivity order\nu::Any\nx-component of advecting flow\nv::Any\ny-component of advecting flow\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams-Tuple{Any, Any, Any, Any, Function, Function, Any}","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","text":"ConstDiffSteadyFlowParams(η, κ, κh, nκh, u::Function, v::Function, grid)\nConstDiffSteadyFlowParams(η, κ, u, v, grid)\n\nThe constructor for the params struct for constant diffusivity problem and steady flow.\n\n\n\n\n\n","category":"method"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.TurbulentFlowParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.TurbulentFlowParams","text":"struct TurbulentFlowParams{T} <: AbstractTurbulentFlowParams\n\nA struct containing the parameters for a constant diffusivity problem with flow obtained from a GeophysicalFlows.MultiLayerQG problem.\n\nη::Any\nisotropic horizontal diffusivity coefficient\nκ::Any\nisotropic vertical diffusivity coefficient\nκh::Any\nisotropic hyperdiffusivity coefficient\nnκh::Int64\nisotropic hyperdiffusivity order\nnlayers::Int64\nnumber of layers in which the tracer is advected-diffused\ntracer_release_time::Any\nflow time prior to releasing tracer\nMQGprob::FourierFlows.Problem\nMultiLayerQG.Problem to generate the advecting flow\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.TurbulentFlowParams-NTuple{4, Any}","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.TurbulentFlowParams","text":"TurbulentFlowParams(η, κ, tracer_release_time, MQGprob)\n\nThe constructor for the params for a constant diffusivity problem with flow obtained from a GeophysicalFlows.MultiLayerQG problem.\n\n\n\n\n\n","category":"method"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.Vars","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.Vars","text":"struct Vars{Aphys, Atrans} <: AbstractVars\n\nThe variables for TracerAdvectionDiffussion problems.\n\nc\ntracer concentration\ncx\ntracer concentration x-derivative, ∂c/∂x\ncy\ntracer concentration y-derivative, ∂c/∂y\nch\nFourier transform of tracer concentration\ncxh\nFourier transform of tracer concentration x-derivative, ∂c/∂x\ncyh\nFourier transform of tracer concentration y-derivative, ∂c/∂y\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.Vars-Union{Tuple{T}, Tuple{Dev}, Tuple{Dev, AbstractGrid{T, A, Alias} where {A, Alias}}} where {Dev, T}","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.Vars","text":"Vars(dev, grid)\n\nReturn the variables vars for a constant diffusivity problem on grid and device dev.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"man/functions/#Functions-exported-from-PassiveTracerFlows:","page":"Functions","title":"Functions exported from PassiveTracerFlows:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#Functions-exported-from-TracerAdvectionDiffusion:","page":"Functions","title":"Functions exported from TracerAdvectionDiffusion:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.Problem-Tuple{Any, FourierFlows.Problem}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.Problem","text":"Problem(dev, MQGprob; parameters...)\n\nConstruct a constant diffusivity problem on device dev using the flow from a GeophysicalFlows.MultiLayerQG problem.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.set_c!-Tuple{Any, Union{PassiveTracerFlows.TracerAdvectionDiffusion.AbstractConstDiffParams, PassiveTracerFlows.TracerAdvectionDiffusion.AbstractSteadyFlowParams}, Any, Any, Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.set_c!","text":"set_c!(sol, params::Union{AbstractConstDiffParams, AbstractSteadyFlowParams}, grid, c)\n\nSet the solution sol as the transform of c and update variables vars.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.set_c!-Union{Tuple{T}, Tuple{Any, PassiveTracerFlows.TracerAdvectionDiffusion.AbstractTurbulentFlowParams, Any, AbstractGrid{T, A, Alias} where {A, Alias}, Any}} where T","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.set_c!","text":"set_c!(sol, params::AbstractTurbulentFlowParams, grid, c)\n\nSet the initial condition for tracer concentration in all layers of a TracerAdvectionDiffusion.Problem that uses a MultiLayerQG flow to  advect the tracer.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!-NTuple{4, Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!","text":"updatevars!(prob)\n\nUpdate the prob.vars in problem prob using the solution prob.sol.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!-Tuple{PassiveTracerFlows.TracerAdvectionDiffusion.AbstractTurbulentFlowParams, Any, Any, Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!","text":"updatevars!(params::AbstractTurbulentFlowParams, vars, grid, sol)\n\nUpdate the varson thegridwith the solution insolfor a problemprob` that is being advected by a turbulent flow.     \n\n\n\n\n\n","category":"method"}]
}
