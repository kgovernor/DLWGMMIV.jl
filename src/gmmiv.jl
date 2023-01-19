# Generalize code for varible number of inputs?? ->i.e. accept list of inputs...
include("config.jl")

function GMM_DLW2(df, betas; all=false, prodF = "CD")
    PHI = df.q
    PHI_LAG = df.phi_lag
    X = [df.k1 df.l1 df.k1l1 df.k2 df.l2]
    X_lag = [df.k_lag df.l_lag df.k_lagl_lag df.k_lag2 df.l_lag2]
    Z = [df.k1 df.l_lag df.kl_lag df.k2 df.l_lag2]

    OMEGA = PHI - X*betas
    OMEGA_lag = PHI_LAG - X_lag*betas
    OMEGA_lag_pol = [df.constant OMEGA_lag]
    OMEGA_lag_polsq = OMEGA_lag_pol'OMEGA_lag_pol
    
    g_b = qr(OMEGA_lag_polsq) \ OMEGA_lag_pol'OMEGA
    g_b[1] = 0
    XI = OMEGA .- OMEGA_lag_pol*g_b
    
    crit = transpose(Z'XI)*(Z'XI)

    if all
        return (crit = crit, diff_crit = diff_crit, XI = XI, g_b = g_b)
    end
    
    return crit
end

function dlwGMMIV(year, plantid, Q, K, L; bstart = zeros(14), prodF="CD", opt="nm", M=[], E=[])
    prodF_options = ["CD", "tl"]
    optimization_options = ["nm", "LBFGS"]
    if prodF ∉ prodF_options | opt ∉ optimization_options
        # TODO - write error function!!! 
        println("error!")
        return
    end

    M = isempty(M) ? zeros(length(K)) : M
    E = isempty(E) ? zeros(length(K)) : E
    
    df = DataConfig(Q, K, L, M, E, year, plantid)
    crit(betas) = GMM_DLW2(df, betas, prodF = prodF)

    # Initialize local optimization model
    opt_parameters = Optim.Options(iterations = 500)
    nm_parameters = Optim.FixedParameters(δ = 0.1)
          
    if opt == "LBFGS"
        p = Optim.optimize(crit, bstart, LBFGS(); autodiff = :forward)
    else    
        p = Optim.optimize(crit, bstart, NelderMead(parameters  = nm_parameters), opt_parameters)
    end
    
    conv_msg = Optim.converged(p)
    beta_names = (:beta_k, :beta_l, :beta_kl, :beta_k2, :beta_l2)
    beta_values = Optim.minimizer(p)
    beta_dlw = (; zip(beta_names, beta_values)...)
    valstart = crit(bstart)
    valend = crit(beta_values)
    ###

    return ((conv_msg = conv_msg, valstart = valstart, valend = valend, beta_dlw = beta_dlw))
end