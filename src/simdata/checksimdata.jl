# Function to check sim data validity 
function sim_data_validity_check(X::AbstractMatrix{<:Real}, X_ω::AbstractMatrix{<:Real}, ω::RealOrVec, params::AbstractParameters; derivative_free::Bool = true, tol::Real = 1e-4)
    funcs = Functional_Forms()

    if (params.prod_params.func != :cd)
        derivative_free = false
    end

    # Use ForwardDiff to check conditions?
    foc_pass, soc_pass = derivative_check(funcs, params, X, X_ω, ω, derivative_free, tol)

    println("\n  First order derivative at optimal L is approximately zero: ", all(foc_pass))
    if !all(foc_pass)
        println("   Simmed data does not pass first order conditions!")
        # for i in findall(.!foc_pass)
        #     println("  foc failed at observation $(i): $(foc[i])")
        # end
    end
  
    println("\n  Second order derivative at optimal L check: ", all(soc_pass)) 
    if !all(soc_pass)
        println("   Simmed data does not pass second order conditions!")
        # for i in findall(.!soc_pass)
        #     println("  soc failed at observation $(i): $(soc[i])")
        # end    
    end
    
    return (foc_pass = foc_pass, soc_pass = soc_pass)
end

# function to check derivative conditions using ForwardDiff
function derivative_check(funcs::AbstractFunctions, params::AbstractParameters, X::AbstractMatrix{<:Real}, X_ω::AbstractMatrix{<:Real}, ω::RealOrVec, derivative_free::Bool, tol::Real)
    n, n_indp = getnuminputs(params)
    ndeps = n - n_indp
    N = size(X, 1)
    prod = params.prod_params
    cost = params.cost_params
    F_prod = funcs.prod[prod.func]
    F_cost = funcs.cost[cost.func](cost)

    foc = Matrix{Real}(undef, N, ndeps)
    soc = Vector{Matrix{Real}}(undef, N)

    deps = n_indp+1:n

    if derivative_free
        F = F_prod(prod, N)
        for i in 1:N
            soc[i] = Matrix(undef, ndeps, ndeps)
        end
        for i in deps
            d1 = i - n_indp
            foc[:, d1] = F(ω, X_ω, X, derivative = [i]) .- F_cost(X, derivative = [i])
            for j in deps
                d2 = j - n_indp
                if i <= j
                    sol = F(ω, X_ω, X, derivative = [i, j]) .- F_cost(X, derivative = [i, j])
                    for k in eachindex(sol)
                        soc[k][d2,d1] = sol[k]
                        if j != i
                            soc[k][d1,d2] = sol[k]
                        end
                    end
                end
            end 
        end
    else
        # Profit Function
        P(ω, X_ω, X) = F_prod(prod)(ω, X_ω, X) - F_cost(X)
        r = DiffResults.HessianResult(zeros(ndeps))

        for i in 1:N
            ForwardDiff.hessian!(r, xdeps -> P(ω[i], X_ω[i, :], [X[i, 1:n_indp]; xdeps]), X[i, deps])

            foc[i, :] = DiffResults.gradient(r)
            soc[i] =  DiffResults.hessian(r)
        end
    end

    foc_pass = abs.(foc) .< tol
    soc_pass = SOC_check.(soc)

    return foc_pass, soc_pass
end

# function to check SOC conditions using ForwardDiff
function SOC_check(H) 
    return all(eigen(H).values .<= 0)
end

