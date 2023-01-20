include("config.jl")

function GMM_DLW2(betas, PHI, PHI_LAG, X, X_lag, Z; all=false)
    OMEGA = PHI - X*betas
    OMEGA_lag = PHI_LAG - X_lag*betas
    OMEGA_lag_pol = [ones(length(OMEGA_lag)) OMEGA_lag]# OMEGA_lag.^2 OMEGA_lag.^3]
    OMEGA_lag_polsq = OMEGA_lag_pol'OMEGA_lag_pol
    
    g_b = qr(OMEGA_lag_polsq) \ OMEGA_lag_pol'OMEGA
    g_b[1] = 0
    XI = OMEGA .- OMEGA_lag_pol*g_b
    
    crit = transpose(Z'XI)*(Z'XI)

    if all
        return (crit = crit, OMEGA = OMEGA, XI = XI, g_b = g_b)
    end
    
    return crit
end

function dlwGMMIV(year, plantid, Q, input1, input2, inputs...; bstart = [], prodF="CD", opt="nm")
    prodF_options = ["CD", "tl"]
    optimization_options = ["nm", "LBFGS"]
    if (prodF ∉ prodF_options) || (opt ∉ optimization_options) 
        # TODO - write error function!!! 
        println("error!")
        return
    end

    inputs = [[input1, input2]; collect(inputs)]

    df = DataConfig(year, plantid, Q, inputs...)
    X_str = ["x"*string(i) for i in eachindex(inputs)]
    X = Matrix(df[:, X_str.*"1"])
    X_lag = Matrix(df[:, X_str.*"_lag"])
    Z = Matrix(df[:, [X_str[1]*"1"; X_str[2:end].*"_lag"]])

    if prodF == "tl"
        X_comb = combinations(X_str, 2)

        find_names = [[join(x_comb, "1")*"1" for x_comb in X_comb]; X_str.*"2"]
        X = hcat(X, Matrix(df[:, find_names]))
        find_names = [[join(x_comb, "_lag")*"_lag" for x_comb in X_comb]; X_str.*"_lag2"]
        X_lag = hcat(X_lag, Matrix(df[:, find_names]))
        find_names = [["x1" ∈ x_comb ? join(x_comb)*"_lag" : join(x_comb, "_lag")*"_lag" for x_comb in X_comb]; X_str[1].*"2"; X_str[2:end].*"_lag2"]
        Z = hcat(Z, Matrix(df[:, find_names]))  

        X_str_temp = [[join(x_comb) for x_comb in X_comb]; X_str.*"2"]
        X_str = [X_str; X_str_temp]

    end
    
    crit(betas) = GMM_DLW2(betas, df.phi, df.phi_lag, X, X_lag, Z)

    # Initialize local optimization model
    if isempty(bstart)
        bstart = zeros(length(X_str))
    elseif length(bstart) != length(X)
        # TODO - write error function!!
        println("error!")
    end
    
    opt_parameters = Optim.Options(iterations = 500)
    nm_parameters = Optim.FixedParameters(δ = 0.1)
          
    if opt == "LBFGS"
        p = Optim.optimize(crit, bstart, LBFGS(); autodiff = :forward)
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