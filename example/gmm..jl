### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 5fe223f9-d03a-4b26-abc2-9e6e37c69a91
begin
	import Pkg
	Pkg.add(url="https://github.com/kgovernor/DLWGMMIV.jl.git")
end

# ╔═╡ 45cebde6-6186-489b-9614-b94079252840
using LinearAlgebra, PlutoUI, Plots

# ╔═╡ 01c7cd66-c991-4cd3-91a7-7f562455eba4
using DLWGMMIV

# ╔═╡ fc9f4ea2-5625-43f3-93ff-b351c9ea6360
begin
	N = 10000
	T = 15
    params = Parameters(
		omega_coefs =[0, 0.8]
	)
	
    by = ["t", "n"]
    yvar = ["Y"]
    xvars1 = ["x1","x2"]
    xvars2 = ["x1","x2"]
    zvars = ["x1","x2_lag"]
end;

# ╔═╡ a8b769e9-2cab-4ad2-bc4e-3c14eb0b88f6
df_main = simfirmdata(N, T, params = params, seed = 1)

# ╔═╡ db3a36a2-5348-4593-ad07-99fc51e21a37
begin
	df = deepcopy(df_main)
	df[:, unique([yvar; xvars1; xvars2])] = log.(df[:, unique([yvar;xvars1;xvars2])])
    df = lag_panel(df, by, ["x2"])
end

# ╔═╡ f707b275-9266-4ab2-b82f-759ae3c067a0
begin
	function gmm_value(df, betas, model)
		XI = f(betas, )
		v = (XI' * Z * Z' * XI) / (nrow(df)^2)
		return v
	end
	V(betas) = gmm_value(df, betas)
end

# ╔═╡ Cell order:
# ╠═45cebde6-6186-489b-9614-b94079252840
# ╠═5fe223f9-d03a-4b26-abc2-9e6e37c69a91
# ╠═01c7cd66-c991-4cd3-91a7-7f562455eba4
# ╠═fc9f4ea2-5625-43f3-93ff-b351c9ea6360
# ╠═a8b769e9-2cab-4ad2-bc4e-3c14eb0b88f6
# ╠═db3a36a2-5348-4593-ad07-99fc51e21a37
# ╠═f707b275-9266-4ab2-b82f-759ae3c067a0
