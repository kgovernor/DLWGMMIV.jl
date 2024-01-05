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
using LinearAlgebra, PlutoUI, Plots, DataFrames, DataFramesMeta

# ╔═╡ 01c7cd66-c991-4cd3-91a7-7f562455eba4
using DLWGMMIV

# ╔═╡ fc9f4ea2-5625-43f3-93ff-b351c9ea6360
begin
	N = 1000
	T = 15
    params = Parameters(
		# omega_coefs = [0, 0.8],
		# cost_exps = [1, 1.15]
	)
	model = ACF_model(
		use_constant = false,
		omega_deg = 3,
		stage2_deg = 2,
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
	seed = 1,
	opt_error = 0
	)

# ╔═╡ db3a36a2-5348-4593-ad07-99fc51e21a37
begin
	df = deepcopy(df_main)
	df[:, unique([yvar; xvars1; xvars2])] = log.(df[:, unique([yvar;xvars1;xvars2])])
    df = lag_panel(df, by, ["x2"])
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
	start, stop = 0, 1
	l = abs(stop - start) * 100
	x1s = range(start, stop, length=l)
	x2s = x1s
	@time Y = [V([x1,x2,0,0,0]) for x1 in x1s, x2 in x2s]
	# @time gbs = [gf([x1,x2,0,0,0])[2] for x1 in x1s, x2 in x2s]
end

begin
	p1 = plot(x1s, log.(Y))
	# p1 = plot(x1s, gbs)
	title!("log of crit")
	xlabel!("βk")

	p1

	# p2 = plot(x1s, x2s, log.(Y),  st=:contour)
	# title!("log of crit, contour")
	# xlabel!("βk")
	# ylabel!("βl")

	# p2
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
