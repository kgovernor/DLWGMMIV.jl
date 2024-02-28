### A Pluto.jl notebook ###
# v0.17.3

# using Markdown
# using InteractiveUtils

# # ╔═╡ 5fe223f9-d03a-4b26-abc2-9e6e37c69a91
# begin
# 	import Pkg
# 	Pkg.add(url="https://github.com/kgovernor/DLWGMMIV.jl.git")
# end

# ╔═╡ 45cebde6-6186-489b-9614-b94079252840
using LinearAlgebra, PlutoUI, Plots, DataFrames, DataFramesMeta, CSV

# ╔═╡ 01c7cd66-c991-4cd3-91a7-7f562455eba4
using DLWGMMIV
include("montecarlo.jl")

# ╔═╡ fc9f4ea2-5625-43f3-93ff-b351c9ea6360
begin
	N = 10000
	T = 10
	seed = 1
	use_constant = false
	omega_deg = 1
	stage2_deg = 2
    params = Parameters(
		# omega_coefs = [0, 0.8],
		# cost_exps = [1, 1.15]
	)
	model = ACF_model(
		use_constant = use_constant,
		omega_deg = omega_deg,
		stage2_deg = stage2_deg,
	)
	
    by = ["t", "n"]
    yvar = ["Y"]
    xvars1 = ["x1","x2"]
    xvars2 = ["x1","x2"]
    zvars = ["x1","x2_lag"]

	vars = (
	    by = ["t", "n"],
		yvar = ["Y"],
		xvars1 = ["x1","x2"],
		xvars2 = [xvars2; DLWGMMIV.poly_vars(xvars2,  model.s2_deg).v],
		zvars = [zvars; DLWGMMIV.poly_vars(zvars,  model.s2_deg).v],
		phivar = "phi"
	)
end;

# ╔═╡ a8b769e9-2cab-4ad2-bc4e-3c14eb0b88f6
df_main = simfirmdata(N, T, params = params, 
	seed = seed,
	opt_error = 0
)

# ╔═╡ db3a36a2-5348-4593-ad07-99fc51e21a37
begin
	df = deepcopy(df_main)
	df = gmmivdf(df, by, yvar, xvars1, xvars2, ["x2"])
	df[:, vars.phivar] = df.Y
	for v in [xvars2, zvars]
        global df = DLWGMMIV.poly_approx_vars(df, model.s2_deg, v)
    end
end

# ╔═╡ f707b275-9266-4ab2-b82f-759ae3c067a0
begin
	Z = Matrix(df[completecases(df), vars.zvars])
	prodest = DLWGMMIV.prodest_method(model.method)
	cache = DLWGMMIV.Cache([], [], [])

	f(betas, gbetas; derivative = []) = prodest(betas, gbetas, df, vars, model.g_B, derivative = derivative)
	gf(betas) = prodest(betas, df, vars, model.g_B, model.use_constant)
	V(betas) = DLWGMMIV.solveXI(cache, betas, f, gf, model, Z);
end

# function sV(betas, f, gf, Z)
#     XI = f(betas, gf(betas))
#     v = (XI' * Z * Z' * XI) / (size(Z, 1)^2)
#     return v
# end
# sV(betas) = sV(betas, f, gf, Z)

begin
	start, stop = -0.5, 1.5
	l = 200
	x1s = range(start, stop, length=l)
	x2s = x1s
	# @time Y = [V([x1,x2,0,0,0]) for x1 in x1s, x2 in x2s]
	# @time gbs = [gf([x1,x2,0,0,0])[2] for x1 in x1s, x2 in x2s]
end

# begin
# 	Bs = [[], [], []]
# 	brange = 0:0.1:1
# 	for b1 in brange, b2 in brange
# 		push!(Bs[end], (b1, b2))
# 		res = dlwGMMIV(
# 			df, by, yvar, xvars1, xvars2, zvars;  
# 			betas = Betas(),
# 			bstart = [b1, b2, 0, 0 ,0] ,
# 			model = model,
# 			skip_stage1 = true,
# 			df_out = true
# 		);

# 		betas = res.r.s2.betas
# 		for i in 1:(length(Bs)-1)
# 			push!(Bs[i], betas[i])
# 		end
# 	end	
# end

begin
	simulator(N, T, seed = -1) = gmmivdf(
		simfirmdata(N, T, params = params, opt_error = 0, seed = seed), 
		by, yvar, xvars1, xvars2, ["x2"]
	)

	gmmivsolver(df, optimizer, bstart) = dlwGMMIV(
			df, by, yvar, xvars1, xvars2, zvars;  
			betas = Betas(),
			bstart = bstart,
			optimizer = optimizer,
			model = model,
			skip_stage1 = true,
			df_out = true,
			lb = [-Inf, -Inf],
    		ub = [Inf, Inf]
	)

	brange = 0:0.1:1
	bstarts = [[b1, b2, 0, 0, 0] for b1 in brange for b2 in brange] # 
	bstart_mc = [[0.5, 0.5, 0, 0, 0]]
	simulator_bstarts(N, T) = simulator(N, T, seed)
# 	sims = 1000
	opt = OptimizationOptimJL.BFGS()
	ptype = "$(stage2_deg)-$(omega_deg)deg$(use_constant ? "wc" : "nc")"
# 	filename_bstarts = "example/bstarts_betas_seed$(seed)_$(ptype).csv"
# 	filename_mc = "example/mc_betas_$(ptype).csv"

# 	@time df_bstarts = montecarlogmm(N, T, simulator_bstarts, gmmivsolver, bstarts; optimizer = opt, simulations = 1, multithread = false, savefile = filename_bstarts)
# 	@time df_mc = montecarlogmm(N, T, simulator, gmmivsolver, bstart_mc; optimizer = opt, simulations = sims, multithread = false, savefile = filename_mc)
	# @time df_mc1 = montecarlogmm(N, T, simulator_bstarts, gmmivsolver, bstart_mc; optimizer = opt, simulations = 1, multithread = false, savefile = "example/seed$(seed)_$(ptype).csv")

end

begin
	# p1 = plot(
	# 	x1s, log.(Y), title = "log of crit", xlabel = "βk", legend =false
	# )

	# # p2 = plot(
	# # 	x1s, x2s, log.(Y), title = "log of crit, contour", xlabel = "βk", ylabel = "βl", st=:contour
	# # )

	# # p3 = plot(
	# # 	x1s, x2s, log.(Y), title = "log of crit, surface", xlabel = "βk", ylabel = "βl", st=:surface
	# # )

	# p4 = plot(
	# 	df_Bs.β_1, df_Bs.β_2, title = "scatter of res diff bstarts", xlabel = "βk", ylabel = "βl", st=:scatter, ms = 2, legend = false
	# )
end

# ╔═╡ Cell order:
# ╠═45cebde6-6186-489b-9614-b94079252840
# ╠═5fe223f9-d03a-4b26-abc2-9e6e37c69a91
# ╠═01c7cd66-c991-4cd3-91a7-7f562455eba4
# ╠═fc9f4ea2-5625-43f3-93ff-b351c9ea6360
# ╠═a8b769e9-2cab-4ad2-bc4e-3c14eb0b88f6
# ╠═db3a36a2-5348-4593-ad07-99fc51e21a37
# ╠═20576708-e97a-406c-894b-829d1f455283
# ╠═f707b275-9266-4ab2-b82f-759ae3c067a0
