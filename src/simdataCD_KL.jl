# This is the alternative function for simulating Cobb Douglas data for K and L only. (Should be faster than simdata.jl function)
function sim_data_solved_L_CD(N, T; omega = 1, seed = -1, alpha_k = 0.1, alpha_l = 0.25, gamma_l = 0.15, rho0 = 0, rho1 = 0.8, rho2 = 0.8, rho3 = 0.8, lnK_mean = 0, lnK_var = 1)  
    if seed >= 0
        Random.seed!(seed)
    end
    
    # Parameters
    α_k, α_l =  alpha_k, alpha_l # Cobb-Douglas params
    γ_l = gamma_l # Cost function params
    ρ0, ρ1 = rho0, rho1 # Productivity params
    σ_ω = omega # TFP shock variance param

    params = (; α_k, α_l, γ_l, ρ0, ρ1, σ_ω) # Store parameters

    # Initialize dataframe
    firm_decision_df = DataFrame()

    # Solve for L
    solve_L(K, ω) = ((α_l/(1+γ_l)) * exp(ω) * (K ^ α_k)) ^ (1/(1+γ_l-α_l))
     
    # Initialize ω_it and TFP shock distribution
    ω_it = rand(Normal(0, σ_ω), N)
    TFP_shock_dist = Normal(0, σ_ω)
    ξ = zeros(N)
    TFP_shock = ξ

    # Initialize omega and K
    K = exp.(rand(Normal(lnK_mean,lnK_var), N*(T+1)))
    omega = ω_it

    for t in 1:T
        # Add periodic TFP shock for each firm
        TFP_shock = rand(TFP_shock_dist, N)
        ω_it = ρ1.*ω_it .+ TFP_shock
        omega = [omega; ω_it]       
        # Save ξ (TFP shock)
        ξ = [ξ; TFP_shock]
    end

    L_opt = solve_L.(K, omega)
    C_l_opt = L_opt .^ (1+γ_l)
    S_opt = α_k .* log.(K) .+ α_l .* log.(L_opt)
    Y_opt = exp.(S_opt .+ omega)
    P_opt = Y_opt .- C_l_opt
    W = C_l_opt ./ L_opt
    SL = C_l_opt ./ (C_l_opt .+ K)
    SLy = C_l_opt ./ Y_opt

    # Save results to DataFrame
    firm_decision_df = DataFrame(time = repeat(0:T, inner = N), firm = repeat(1:N, outer = (T+1)), K = K, L = L_opt, S = S_opt, Y = Y_opt, P = P_opt, W = W, SL = SL, SLy = SLy, omega_i = omega, XI = ξ)

    # Summary
    vc = sim_data_validity_check_solved_L_CD(firm_decision_df, params)
    total_obs = nrow(firm_decision_df)
    println("\n=======================\n")
    println("SUMMARY:")
    println("\t$(round(100*count(vc.foc_pass)/total_obs, digits = 2))% of observations passed first order conditions.")
    println("\t$(round(100*count(vc.soc_pass)/total_obs, digits = 2))% of observations passed second order conditions.")
    println("\n=======================\n")
    
    return (df=firm_decision_df, params=params, derivative_check = vc)
end
