function solver(df::DataFrame, n::Int, n_indp::Int, F_prod::Function, F_cost::Function, xstart::Real = 1000)   
    n_dep = n-n_indp
    X = Matrix(df[:, genxvars(n_indp)])
    X_ω = Matrix(df[:, names(df, r"OMEGA_")])

    X_opt = Matrix(undef, nrow(df), n)
    if !isempty(X)
        X_opt[:, 1:n_indp] = X[:, 1:n_indp]
    end
    Y_opt = Vector{Float64}(undef, 0)
    termination = Vector{String}(undef, 0)

    nvars = 2n+1

    function F_prod_solve(vars...)
        vars = collect(vars)
        ind = (omega = 1, X_ω = 2:(n+1), X = (n+2):(nvars))

        ωs = vars[ind.omega]
        X_ωs = vars[ind.X_ω]
        Xs = vars[ind.X]
        
        return F_prod(ωs, X_ωs, Xs)
    end

    ## Ipopt Model ##
    # Initialize model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, Xdep[1:n_dep] >= 0, start = xstart)

    @variable(model, omega[i = 1:1] in Parameter(i))
    @variable(model, Xomega[i = 1:n] in Parameter(i))
    @variable(model, Xindp[i = 1:n_indp] in Parameter(i))

    @operator(model, F, nvars, F_prod_solve)
    @operator(model, Fc, n, (x...) -> F_cost(collect(x)))

    @expression(model, Y, F(omega[1], Xomega..., Xindp..., Xdep...))
    @expression(model, TC, Fc(Xindp..., Xdep...))
    @expression(model, P, Y - TC )

    @objective(model, Max, P)
    
    for i in 1:nrow(df)
        set_parameter_value(omega[1], df.OMEGA[i])
        for j in 1:n
            if j <= n_indp
                set_parameter_value(Xindp[j], X[i, j])
            end 
            set_parameter_value(Xomega[j], X_ω[i, j])
        end

        # Run firm decision model 
        JuMP.optimize!(model)  
        xvals = value.(model[:Xdep]) 
        for j in 1:n_dep
            X_opt[i, n_indp+j] = xvals[j] 
        end
        push!(Y_opt, value(model[:Y]))
        push!(termination, string(termination_status(model)))
    end

    return FirmSolution(Y_opt, X_opt, termination)
end