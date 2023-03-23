include("config.jl")

################
# Main methods #
################


"""
    dlwGMMIV(year, plantid, Q, inputs...; <keyword arguments>)

Returns results for estimated betas from GMM IV for industry's production function `prodF` given panel data with `year`, `plantid`, `inputs...` (with independent/dynamic inputs listed first), and output, `Q`.
 
Results are returned as a `NamedTuple`.

# Arguments
- `num_indp_inputs::Integer=1`: the number of independent/dynamic inputs.
- `bstart::Vector{Float64}`: the starting beta values for optimization. Defaults to zeros.
- `δ_nm::Float64`=0.1: the Nelder-Mead shrink step parameter, delta.
- `max_iters::Integer=500`: the maximum number of iterations ran in optimization.

# Configurable Options
- `prodF::String`: the production function to estimate. Default is `"CD"`, Cobb-Douglas; other options include `"tl"`, TransLog.
- `opt::String`: the optimization method to use. Default is `"nm"`, Nelder-Mead; other options include `"LBFGS"`.

# Examples
```jldoctest
julia> using DLWGMMIV

julia> df = DLWGMMIV.sim_data(20000, 10, input_names = ["K", "L"], prod_params = [0.1, 0.25]).df;
[...]

julia> data = [df.time, df.firm, df.Y, df.K, df.L];

julia> res = dlwGMMIV(data...);

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 1.074768000861536e-9

betas = (beta_x1 = 0.10230132568110045, beta_x2 = 0.26251910427769287)

julia> res = dlwGMMIV(data..., opt = "LBFGS");

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 3.6354547893764317e-19

betas = (beta_x1 = 0.10230132472854156, beta_x2 = 0.2625190996231563)

julia> res = dlwGMMIV(data..., opt = "LBFGS", prodF = "tl");

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 3.1089429872638027e-15

betas = (beta_x1 = 0.11394854106849515, beta_x2 = 1.9222327345314836, beta_x1x2 = -0.5175358152780514, beta_x12 = -0.04266689698177511, beta_x22 = 0.00703394572086843)
```
"""
function dlwGMMIV(year, plantid, Q, inputs...; num_indp_inputs = 1, bstart = [], prodF="CD", opt="nm", δ_nm = 0.1, max_iters = 500, use_constant = "")
    prodF_options = ["CD", "tl"]
    optimization_options = ["nm", "LBFGS"]
    if (prodF ∉ prodF_options) || (opt ∉ optimization_options) 
        # TODO - write error function!!! 
        println("error!")
        return
    end

    inputs = collect(inputs)

    df = DataConfig(year, plantid, Q, inputs...)
    X_str, X, X_lag, Z = gen_inputset(df, num_indp_inputs, length(inputs), prodF, use_constant)
    
    crit(betas) = GMM_DLW2(betas, df.phi, df.phi_lag, X, X_lag, Z, use_constant)

    # Initialize local optimization model
    bsize = length(X_str)
    bstart = isempty(bstart) ? zeros(bsize) : length(bstart) < bsize ? [bstart; zeros(bsize - length(bstart))] : bstart[begin:bsize]
    
    opt_parameters = Optim.Options(iterations = max_iters)
    nm_parameters = Optim.FixedParameters(δ = δ_nm)
          
    if opt == "LBFGS"
        p = Optim.optimize(crit, bstart, LBFGS(), opt_parameters; autodiff = :forward)
    elseif opt == "nm"   
        p = Optim.optimize(crit, bstart, NelderMead(parameters  = nm_parameters), opt_parameters)
    end
    
    conv_msg = Optim.converged(p)
    beta_names = Symbol.("beta_".*X_str)
    beta_values = Optim.minimizer(p)
    beta_dlw = (; zip(beta_names, beta_values)...)
    valstart = crit(bstart)
    valend = crit(beta_values)

    other_results = GMM_DLW2(beta_values, df.phi, df.phi_lag, X, X_lag, Z, use_constant, all = true)

    return ((conv_msg = conv_msg, valstart = valstart, valend = valend, beta_dlw = beta_dlw, other_results = other_results))
end


################################################################################################################################################
#################
# Other methods #
#################

### Objective Function ###
# GMM IV - objective function
function GMM_DLW2(betas, PHI, PHI_LAG, X, X_lag, Z, use_constant; all=false)
    OMEGA = PHI - X*betas
    OMEGA_lag = PHI_LAG - X_lag*betas
    #OMEGA_lag_pol = [ones(length(OMEGA_lag)) OMEGA_lag]# OMEGA_lag.^2 OMEGA_lag.^3] # To consider adding!
    OMEGA_lag_pol = OMEGA_lag
    if use_constant in ["omega", "notZ", "all"]
        OMEGA_lag_pol = [ones(length(OMEGA_lag)) OMEGA_lag_pol]
    end
    OMEGA_lag_polsq = OMEGA_lag_pol'OMEGA_lag_pol
    
    #g_b = qr(OMEGA_lag_polsq) \ OMEGA_lag_pol'OMEGA # If above is added
    g_b = OMEGA_lag_polsq \ OMEGA_lag_pol'OMEGA
    #g_b[1] = 0 # May not be needed depending on if constant is added or not. Issues with constant.
    XI = OMEGA .- OMEGA_lag_pol*g_b
    
    crit = transpose(Z'XI)*(Z'XI)

    if all
        return (crit = crit, OMEGA = OMEGA, XI = XI, g_b = g_b)
    end
    
    return crit
end

### Production Function Inputs ###
# Generate input set for GMM IV
function gen_inputset(df, num_indp_inputs, num_inputs, prodF, use_constant)
    X_str = ["x"*string(i) for i in 1:num_inputs]
    X = Matrix(df[:, X_str.*"1"])
    X_lag = Matrix(df[:, X_str.*"_lag"])
    Z = Matrix(df[:, [X_str[begin:num_indp_inputs].*"1"; X_str[(num_indp_inputs+1):end].*"_lag"]])

    if prodF == "tl"
        X_str, X, X_lag, Z = add_inputset_tl(df, num_indp_inputs, X_str, X, X_lag, Z) # input set for translog production function estimation
    end

    if !isempty(use_constant)
        constant = ones(size(X,1))
        if use_constant in ["X", "notZ", "all"]
            X_str = ["constant"; X_str]
            X = [constant X]
            X_lag = [constant X_lag]
        end
        if use_constant in ["Z", "all"]
            Z = [constant Z]
        end
    end

    inputset = (X_str, X, X_lag, Z) # input set for Cobb Douglas production function estimation

    return inputset
end

function add_inputset_tl(df, num_indp_inputs, X_str, X, X_lag, Z)
    X_indp = X_str[1:num_indp_inputs]
    X_comb = combinations(X_str, 2)
    tl_Zinteractions = [length(X_indp ∩ x_comb) == 1 ? join(x_comb)*"_lag" : length(X_indp ∩ x_comb) == 2 ? join(x_comb) : join(x_comb, "_lag")*"_lag" for x_comb in X_comb]

    find_names = [[join(x_comb, "1")*"1" for x_comb in X_comb]; X_str.*"2"]
    X = hcat(X, Matrix(df[:, find_names]))
    find_names = [[join(x_comb, "_lag")*"_lag" for x_comb in X_comb]; X_str.*"_lag2"]
    X_lag = hcat(X_lag, Matrix(df[:, find_names]))
    find_names = [tl_Zinteractions; X_indp.*"2"; X_str[(num_indp_inputs+1):end].*"_lag2"]
    Z = hcat(Z, Matrix(df[:, find_names]))  

    X_str_temp = [[join(x_comb) for x_comb in X_comb]; X_str.*"2"]
    X_str = [X_str; X_str_temp]

    inputset_tl = (X_str, X, X_lag, Z)

    return inputset_tl
end
