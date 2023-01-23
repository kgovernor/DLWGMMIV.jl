# Function to check sim data validity 
function sim_data_validity_check(df, params, funcs, input_params; use_FDiff = false, tol = 1e-4)
    X = df[:, input_params.input_names]
    ω = df.omega_i

    if (params.prodF != "CD") || (input_params.num_inputs > 2)
        use_FDiff == true
    end

    # Profit Function
    S, TC = funcs 
    P(ω, X) = exp(S(X...) + ω) - TC(X...)

    foc_pass = []
    soc_pass = []

    # Use ForwardDiff to check conditions?
    if use_FDiff 
        foc_pass, soc_pass = derivative_check(P, X, ω, input_params.num_indp_inputs, tol)
    else
        foc_pass, soc_pass = derivative_check_CD_2(X[:, 1], X[:, 2], ω, params.prod_params[1], params.prod_params[2], params.cost_params[2], tol)
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

# function to check derivative conditions using ForwardDiff
function derivative_check(P, X, ω, num_indp_inputs, tol)
    num_obs = size(X)[1]
    foc_pass = []
    soc_pass = []

    for n in 1:num_obs
        Xindp = collect(X[n, begin:num_indp_inputs])
        Xdep = collect(X[n, (num_indp_inputs+1):end])

        result = DiffResults.HessianResult(Xdep)
        result = ForwardDiff.hessian!(result,x -> P(ω[n],[Xindp; x]), Xdep)

        foc_pass = [foc_pass; all(abs.(DiffResults.gradient(result)) .< tol)]
        soc_pass = [soc_pass; SOC_check(DiffResults.hessian(result))]
    end

    return foc_pass, soc_pass
end

# function to check SOC conditions using ForwardDiff
function SOC_check(H) 
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

# function to check derivative conditions without using ForwardDiff - (for CobbDouglas 2 inputs only)
function derivative_check_CD_2(K, L, ω, α_k, α_l, γ_l, tol)
    dP_dl(K,L,ω) = α_l*(K^α_k)*(L^(α_l-1))*exp(ω) - (1+γ_l)*(L^γ_l)
    d2P_dl2(K,L,ω) = (α_l - 1)*α_l*(K^α_k)*(L^(α_l-2))*exp(ω) - γ_l*(1+γ_l)*(L^(γ_l-1))

    foc_pass = abs.(dP_dl.(K,L,ω)) .< tol
    soc_pass = d2P_dl2.(K,L,ω) .< 0
    
    return foc_pass, soc_pass
end

######################################################################
### MISC ###

# Function to check sim data validity for sim_data_solved_L_CD
function sim_data_validity_check_solved_L_CD(df, params; use_FDiff = false, tol = 1e-4)
    # Parameters
    α_k, α_l = params.α_k, params.α_l # Cobb-Douglas params
    γ_l = params.γ_l # Cost function params

    # Data to test conditions
    K, L, ω = df.K, df.L, df.omega_i

    # Profit Function
    P(ω,x) = (x[1]^α_k)*(x[2]^α_l)*exp(ω) - (x[2]^(1+γ_l))

    # Check derivatives without ForwardDiff
    foc_pass, soc_pass = derivative_check_CD_2(K, L, ω, α_k, α_l, γ_l, tol)

    # Use ForwardDiff to check conditions
    if use_FDiff 
        foc_pass, soc_pass = derivative_check(P, [K L], ω, 1, tol)
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