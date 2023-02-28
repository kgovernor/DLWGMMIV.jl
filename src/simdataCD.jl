# This is the alternative function for simulating Cobb Douglas data for K and L only. (Should be faster than simdata.jl function)
function sim_data_solved_L_CD(N, T; omega = 1, seed = -1, alpha_k = 0.1, alpha_l = 0.25, gamma_l = 0.15, rho0 = 0, rho1 = 0.8, rho2 = 0, rho3 = 0, lnK_mean = 0, lnK_var = 1)  
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


function solve_CD(indp_inputs, num_inputs, num_indp_inputs, prod_params, cost_params, ω)

    A = repeat(prod_params[(num_indp_inputs+1):num_inputs]',(num_inputs-num_indp_inputs),1)
    for i in axes(A)[1]
        A[i,i] = A[i,i] - 1 - cost_params[num_indp_inputs+i]
    end

    b = [log(1+cost_params[i]) - log(prod_params[i]) - ω - sum(prod_params[j]*log(indp_inputs[j]) for j in 1:num_indp_inputs) for i in (num_indp_inputs+1):num_inputs]

    prob = LinearProblem(A, b)
    sol = solve(prob).u

    solution = exp.(sol)

    return solution
end


# This is the alternative function for simulating Cobb Douglas data. (Should be faster than simdata.jl function)
function sim_data_CD(N, T; num_inputs = 2, num_indp_inputs = 1,  input_names = ["K", "L"], prod_params = [0.1, 0.25], cost_params = [0, 0.15], omega_params = [0, 0.8, 0, 0], indp_inputs_params = [1], σ_ω = 1, indp_inputs_lnmean = [5], indp_inputs_lnvariance = [1], seed = -1)  
    prodF, costF = "CD", "ce"

    if seed >= 0
        Random.seed!(seed)
    end
    
    check_options(num_inputs, num_indp_inputs, prodF, costF)

    # Generate input names
    input_names = gen_input_names(num_inputs, input_names)
    
    # Set up parameters
    prod_params, cost_params, omega_params, indp_inputs_params = gen_params(num_inputs, num_indp_inputs, input_names, prodF, prod_params, cost_params, omega_params, indp_inputs_params)

    # Functions to Generate Data ##
    S_func, TC_func = gen_prodF_costF(prodF, costF, prod_params, cost_params, num_inputs)

    # Initialize independent inputs, e.g. k, capital
    indp_inputs_lnmean = length(indp_inputs_lnmean) < num_indp_inputs ? [indp_inputs_lnmean; 5*ones(num_indp_inputs - length(indp_inputs_lnmean))] : indp_inputs_lnmean[begin:num_indp_inputs]
    indp_inputs_lnvariance = length(indp_inputs_lnvariance) < num_indp_inputs ? [indp_inputs_lnvariance; 5*ones(num_indp_inputs - length(indp_inputs_lnvariance))] : indp_inputs_lnvariance[begin:num_indp_inputs]
    indp_inputs_distributions = [Normal(indp_inputs_lnmean[i], indp_inputs_lnvariance[i]) for i in num_indp_inputs]
    x_indp = [exp.(rand(indp_inputs_distributions[i], N)) for i in 1:num_indp_inputs]
    
    # Initialize dataframe
    firm_decision_df = DataFrame()
     
    # Initialize ω_it and TFP shock distribution
    ω_it = rand(Normal(0, σ_ω), N)
    TFP_shock_dist = Normal(0, σ_ω)
    ξ = zeros(N)
    TFP_shock = ξ

    # Initialize omega
    omega = ω_it

    for t in 1:T
        # Add periodic TFP shock for each firm
        TFP_shock = rand(TFP_shock_dist, N)
        ω_it .= [ones(length(ω_it)) ω_it ω_it.^2 ω_it.^3]*omega_params .+ TFP_shock
        omega = [omega; ω_it]
        # Update independent inputs
        indp = [x_indp[i][(end-N)+1:end] for i in 1:num_indp_inputs]
        x_indp = [[x_indp[i]; indp[i]*(indp_inputs_params[i]*rand(Normal(1,0.1)))] for i in 1:num_indp_inputs]       
        # Save ξ (TFP shock)
        ξ = [ξ; TFP_shock]
    end

    X = [ i <= num_indp_inputs ? x_indp[i] : [] for i in 1:num_inputs]

    for i in eachindex(omega)
        xindp = [X[j][i] for j in 1:num_indp_inputs]
        sol =  solve_CD(xindp, num_inputs, num_indp_inputs, prod_params, cost_params, omega[i])
        for j in eachindex(sol)
            X[num_indp_inputs+j] = [X[num_indp_inputs+j] ; sol[j]]
        end
    end

    X_opt = (; zip(Symbol.(input_names), X)...)
    TC_opt = TC_func.(X...)
    S_opt = S_func.(X...)
    Y_opt = exp.(S_opt .+ omega)
    P_opt = Y_opt .- TC_opt

    C = []
    rent = []
    share_TC = []
    share_Y = []
    for i in 1:num_inputs
        input = [ i == j ? X[i] : zeros(length(TC_opt)) for j in 1:num_inputs]
        push!(C, TC_func.(input...))
        push!(rent, C[i]./X[i])
        push!(share_TC, C[i]./TC_opt)
        push!(share_Y, C[i]./Y_opt)
    end
    res = (; zip(Symbol.(["C_".*input_names; "rent_".*input_names; "share_TC_".*input_names; "share_Y_".*input_names]), [C; rent; share_TC; share_Y])...)

    # Save results to DataFrame
    firm_decision_df = DataFrame(time = repeat(0:T, inner = N), firm = repeat(1:N, outer = (T+1)), S = S_opt, Y = Y_opt, P = P_opt, TC = TC_opt, omega_i = omega, XI = ξ)
    firm_decision_df = hcat(firm_decision_df, DataFrame(X_opt))
    firm_decision_df = hcat(firm_decision_df, DataFrame(res))

    # Store parameters and functions
    params = (prod_params = prod_params, cost_params = cost_params, omega_params = omega_params, σ_ω = σ_ω, prodF = prodF) 
    funcs = (S_func = S_func, TC_func = TC_func)
    input_params = (input_names = input_names, num_inputs = num_inputs, num_indp_inputs = num_indp_inputs)

    # Summary
    vc = sim_data_validity_check(firm_decision_df, params, funcs, input_params)
    total_obs = nrow(firm_decision_df)
    println("\n=======================\n")
    println("SUMMARY:")
    println("\t$(round(100*count(vc.foc_pass)/total_obs, digits = 2))% of observations passed first order conditions.")
    println("\t$(round(100*count(vc.soc_pass)/total_obs, digits = 2))% of observations passed second order conditions.")
    println("\n=======================\n")
    
    return (df=firm_decision_df, params=params, funcs = funcs, input_params = input_params, derivative_check = vc)
end

