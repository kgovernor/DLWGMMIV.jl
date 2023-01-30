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
"""
function dlwGMMIV(year, plantid, Q, inputs...; num_indp_inputs = 1, bstart = [], prodF="CD", opt="nm", δ_nm = 0.1, max_iters = 500)
    prodF_options = ["CD", "tl"]
    optimization_options = ["nm", "LBFGS"]
    if (prodF ∉ prodF_options) || (opt ∉ optimization_options) 
        # TODO - write error function!!! 
        println("error!")
        return
    end

    inputs = collect(inputs)

    df = DataConfig(year, plantid, Q, inputs...)
    X_str, X, X_lag, Z = gen_inputset(df, num_indp_inputs, inputs, prodF)
    
    crit(betas) = GMM_DLW2(betas, df.phi, df.phi_lag, X, X_lag, Z)

    # Initialize local optimization model
    if isempty(bstart)
        bstart = zeros(length(X_str))
    elseif length(bstart) != length(X)
        # TODO - write error function!!
        println("error!")
    end
    
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

    other_results = GMM_DLW2(beta_values, df.phi, df.phi_lag, X, X_lag, Z, all = true)

    return ((conv_msg = conv_msg, valstart = valstart, valend = valend, beta_dlw = beta_dlw, other_results = other_results))
end


################################################################################################################################################
#################
# Other methods #
#################

### Objective Function ###
# GMM IV - objective function
function GMM_DLW2(betas, PHI, PHI_LAG, X, X_lag, Z; all=false)
    OMEGA = PHI - X*betas
    OMEGA_lag = PHI_LAG - X_lag*betas
    #OMEGA_lag_pol = [ones(length(OMEGA_lag)) OMEGA_lag]# OMEGA_lag.^2 OMEGA_lag.^3] # To consider adding!
    OMEGA_lag_pol = OMEGA_lag
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
function gen_inputset(df, num_indp_inputs, inputs, prodF)
    X_str = ["x"*string(i) for i in eachindex(inputs)]
    X = Matrix(df[:, X_str.*"1"])
    X_lag = Matrix(df[:, X_str.*"_lag"])
    Z = Matrix(df[:, [X_str[begin:num_indp_inputs].*"1"; X_str[(num_indp_inputs+1):end].*"_lag"]])

    inputset = (X_str, X, X_lag, Z) # input set for Cobb Douglas production function estimation

    if prodF == "tl"
        inputset = add_inputset_tl(df, num_indp_inputs, inputset...) # input set for translog production function estimation
    end

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
