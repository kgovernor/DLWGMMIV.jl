# This is the main function for simulating data. 
## TO DO - add M and E?
function sim_data_CD(N, T; omega = 1, seed = 5, alpha_k = 0.1, alpha_l = 0.25, gamma_l = 0.15, rho0 = 0, rho1 = 0.8, start_L = 1000, kwargs...)
    #Random.seed!(seed)
   
    # Parameters
    ## TO DO - can get paramameter from kwargs...?
    α_k, α_l, α_m, α_e =  alpha_k, alpha_l, 0, 0 # Cobb-Douglas params
    γ_l, γ_m, γ_e = gamma_l, 0, 0 # Cost function params
    ρ0, ρ1, ρ2, ρ3 = rho0, rho1, 0, 0 # Productivity params
    σ_ω = omega # TFP shock variance param

    params = (; α_k, α_l, α_m, α_e, γ_l, γ_m, γ_e, ρ0, ρ1, ρ2, ρ3, σ_ω) # Store parameters

    # Initialize dataframe
    firm_decision_df = DataFrame()
     
    # Initialize model
    model = Model(Ipopt.Optimizer)
    set_silent(model)
  
    ## Set variables
    @variable(model, L >= 0, start = start_L)
    @NLparameter(model, p[i = 1:2] == i)
    
    ## defining labor costs
    @NLexpression(model, C_l, L^(1+γ_l) )
    
    ## defining production function
    @NLexpression(model, S, α_k*log(p[1]) + α_l*log(L) )
    
    @NLexpression(model, Y, exp(S + p[2]) )
    
    ## defining profit
    @NLexpression(model, P, Y - C_l )
    
    ## defining objective
    @NLobjective(model, Max, P)
    
    
    # Initialize ω_it and TFP shock distribution
    ω_it = rand(Normal(0, 1), N)
    TFP_shock_dist = Normal(0, σ_ω)
    ξ = zeros(N)
    TFP_shock = ξ

    for t in 0:T
        # Initialize k_it
        k_it = exp.(rand(Normal(0,1), N))

        for n in 1:N
            set_value(p[1], k_it[n])
            set_value(p[2], ω_it[n])
            
            # Run firm decision model
            JuMP.optimize!(model)
            
            K, L_opt, C_l_opt, Y_opt, S_opt, P_opt = value(p[1]), value(L), value(C_l), value(Y), value(S), value(P)
            W = C_l_opt/L_opt
            SL = C_l_opt/(C_l_opt + K)
            SLy = C_l_opt/Y_opt
            termination = string(termination_status(model))

            # Save results in dataframe
            firm_decision = ( (omega_i=value(p[2]), start_L=start_L, L=L_opt, K=K, W=W, C_l=C_l_opt, Y=Y_opt, SL=SL, SLy=SLy, S=S_opt, P=P_opt, termination=termination))       
            push!(firm_decision_df, firm_decision; cols = :union)
        end
        
        # Save ξ (TFP shock)
        if t > 0
            ξ = [ξ; TFP_shock]
        end

        # Add periodic TFP shock for each firm
        TFP_shock = rand(TFP_shock_dist, N)
        ω_it .= ρ1.*ω_it .+ TFP_shock
    end

    # Save results to DataFrame
    firm_decision_df = hcat(DataFrame(time = repeat(0:T, inner = N), firm = repeat(1:N, outer = (T+1)), XI = ξ), firm_decision_df)
    
    return (df=firm_decision_df, params=params)
end


# This is the alternative function for simulating data for K and L only. (Should be faster than the one above)
function sim_data_solved_L(N, T; omega = 1, seed = 5, alpha_k = 0.1, alpha_l = 0.25, gamma_l = 0.15, rho0 = 0, rho1 = 0.8)
    #Random.seed!(seed)
   
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
    ω_it = rand(Normal(0, 1), N)
    TFP_shock_dist = Normal(0, σ_ω)
    ξ = zeros(N)
    TFP_shock = ξ

    # Initialize omega and K
    K = exp.(rand(Normal(0,1), N*(T+1)))
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

    return (df=firm_decision_df, params=params)
end


# Function to check sim data validity 
function sim_data_validity_check_KL(df, params; use_FDiff = false, tol = 1e-4)
    # Parameters
    α_k, α_l = params.α_k, params.α_l # Cobb-Douglas params
    γ_l = params.γ_l # Cost function params

    # Data to test conditions
    K, L, ω = df.K, df.L, df.omega_i

    # Profit Function and derivatives
    P(K,L,ω) = (K^α_k)*(L^α_l)*exp(ω) - (L^(1+γ_l))
    dP_dl(K,L,ω) = α_l*(K^α_k)*(L^(α_l-1))*exp(ω) - (1+γ_l)*(L^γ_l)
    d2P_dl2(K,L,ω) = (α_l - 1)*α_l*(K^α_k)*(L^(α_l-2))*exp(ω) - γ_l*(1+γ_l)*(L^(γ_l-1))

    # Satisfies first order and second order conditions?
    foc_pass = (dP_dl.(K,L,ω)).^2 .< tol
    soc_pass = d2P_dl2.(K,L,ω) .< 0

    # Use ForwardDiff to check conditions
    if use_FDiff 
        function SOC_check(K,L,ω) # function to check SOC conditions using ForwardDiff
            H = ForwardDiff.hessian(x ->P(K,x[1],ω), [L])
            if H[1,1] < 0  # SOC
                if (length(H[1]) > 1) 
                    if (det(H) <= 0)
                        return false
                    end
                end
                return true
            else
                return false
            end
        end
        foc_pass = [sum(ForwardDiff.gradient(x ->P(K[i],x[1],ω[i]), [L[i]]).^2) for i in 1:length(K)] .< tol
        soc_pass = SOC_check.(K,L,ω)
    end

    println("\n  First order derivative at optimal L is approximately zero: ", all(foc_pass))
    if !all(foc_pass)
        println("   Simmed data does not pass first order conditions!")
    end
  
    println("  Second order derivative at optimal L check: ", all(soc_pass)) 
    if !all(soc_pass)
        println("   Simmed data does not pass second order conditions!")
    end
    
    return (foc_pass = foc_pass, soc_pass = soc_pass)
end