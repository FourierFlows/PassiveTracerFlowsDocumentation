var documenterSearchIndex = {"docs":
[{"location":"modules/traceradvectiondiffusion/#TracerAdvectionDiffusion-Module","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/#Basic-Equations","page":"TracerAdvectionDiffusion Module","title":"Basic Equations","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"This module solves the advection diffusion equation for a passive tracer concentration  c(x y t) in two-dimensions by an advecting flow bmu(x y t):","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t c + bmu bmcdot bmnabla c = underbraceeta partial_x^2 c + kappa partial_y^2 c_textrmdiffusivity + underbracekappa_h (-1)^n_h nabla^2n_hc_textrmhyper-diffusivity ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmu = (u v) is the two-dimensional advecting flow, eta the x-diffusivity and kappa is the y-diffusivity. If eta is not defined then the code uses isotropic diffusivity, i.e., eta partial_x^2 c + kappa partial_y^2 c mapsto kappa nabla^2. The advecting flow could be either compressible or incompressible. ","category":"page"},{"location":"modules/traceradvectiondiffusion/#Implementation","page":"TracerAdvectionDiffusion Module","title":"Implementation","text":"","category":"section"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"The equation is time-stepped forward in Fourier space:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"partial_t widehatc = - widehatbmu bmcdot bmnabla c - left (eta k_x^2 + kappa k_y^2) + kappa_h bmk^2nu_h right widehatc ","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"where bmk = (k_x k_y).","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"Thus:","category":"page"},{"location":"modules/traceradvectiondiffusion/","page":"TracerAdvectionDiffusion Module","title":"TracerAdvectionDiffusion Module","text":"beginaligned\nL  = -eta k_x^2 - kappa k_y^2 - kappa_h bmk^2nu_h  \nN(widehatc) = - mathrmFFT(u partial_x c + v partial_y c) \nendaligned","category":"page"},{"location":"#PassiveTracerFlows.jl-Documentation","page":"Home","title":"PassiveTracerFlows.jl Documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows.jl is a collection of modules which leverage the  FourierFlows.jl framework to solve for advection-diffusion problems on periodic domains.","category":"page"},{"location":"","page":"Home","title":"Home","text":"info: Unicode\nOftentimes unicode symbols are used in modules for certain variables or parameters. For  example, κ is commonly used to denote the diffusivity, or ∂ is used  to denote partial differentiation. Unicode symbols can be entered in the Julia REPL by  typing, e.g., \\kappa or \\partial followed by the tab key.Read more about Unicode symbols in the  Julia Documentation.","category":"page"},{"location":"#Developers","page":"Home","title":"Developers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PassiveTracerFlows is currently being developed by Navid C. Constantinou and Gregory L. Wagner.","category":"page"},{"location":"#Cite","page":"Home","title":"Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The code is citable via zenodo, doi:10.5281/zenodo.2535983.","category":"page"},{"location":"man/types/#Private-types","page":"Private types","title":"Private types","text":"","category":"section"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#Private-types-in-module-PassiveTracerFlows:-2","page":"Private types","title":"Private types in module PassiveTracerFlows:","text":"","category":"section"},{"location":"man/types/","page":"Private types","title":"Private types","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPublic = false\nOrder = [:type]","category":"page"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffParams","text":"ConstDiffParams(eta, kap, kaph, nkaph, u, v)\nConstDiffParams(eta, kap, u, v)\n\nReturns the params for constant diffusivity problem with time-varying flow.\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.ConstDiffSteadyFlowParams","text":"ConstDiffSteadyFlowParams(eta, kap, kaph, nkaph, u, v, g)\nConstDiffSteadyFlowParams(eta, kap, u, v, g)\n\nReturns the params for constant diffusivity problem with time-steady flow.\n\n\n\n\n\n","category":"type"},{"location":"man/types/#PassiveTracerFlows.TracerAdvectionDiffusion.Vars-Union{Tuple{T}, Tuple{Dev}, Tuple{Dev,FourierFlows.AbstractGrid{T,A} where A}} where T where Dev","page":"Private types","title":"PassiveTracerFlows.TracerAdvectionDiffusion.Vars","text":"Vars(g)\n\nReturns the vars for constant diffusivity problem on grid g.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"man/functions/#Functions-exported-from-PassiveTracerFlows:","page":"Functions","title":"Functions exported from PassiveTracerFlows:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#Functions-exported-from-TracerAdvectionDiffusion:","page":"Functions","title":"Functions exported from TracerAdvectionDiffusion:","text":"","category":"section"},{"location":"man/functions/","page":"Functions","title":"Functions","text":"Modules = [PassiveTracerFlows.TracerAdvectionDiffusion]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.set_c!-Tuple{Any,Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.set_c!","text":"set_c!(prob, c)\n\nSet the solution sol as the transform of c and update variables v on the grid g.\n\n\n\n\n\n","category":"method"},{"location":"man/functions/#PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!-Tuple{Any}","page":"Functions","title":"PassiveTracerFlows.TracerAdvectionDiffusion.updatevars!","text":"updatevars!(prob)\n\nUpdate the vars in v on the grid g with the solution in sol.\n\n\n\n\n\n","category":"method"}]
}
