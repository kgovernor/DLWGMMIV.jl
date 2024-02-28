function poly_autoreg(x::RealOrVec, coefs::Vector{<:Real})
    x_new = zeros(length(x))
    for i in 1:length(coefs)
        @. x_new += coefs[i] * (x ^ (i-1))
    end

    return x_new
end

function genvarnames(str::String, stop::Int; start::Int = 1)
    varnames = ["$str$i" for i in start:stop]
    return varnames
end

genxvars(stop; start = 1) = genvarnames("x", stop; start = start)

function sim_data(N::Int, T::Int; data_params::AbstractParameters = Params(), lognormal::Bool = false, add_error::Bool = true)
    p = data_params.i

    ϵ = rand(data_params.err, N*T)
    ϵ = ϵ isa Vector ? ϵ : ϵ' 
    n = size(ϵ, 2)
    data = Matrix{Float64}(undef, size(ϵ, 1), n)

    if isempty(data_params)
        start = lognormal ? ones(n) : zeros(n)
        coefs = zeros(n)
    else
        start = data_params[p.start][p.start]
        coefs = collect(values(data_params, p.coefs, false))
    end

    for t in 1:T
        for i in 1:n
            prev = (N*(t-2)+1):(N*(t-1))
            cur =  (N*(t-1)+1):(N*t)

            if t < 2
                x = start[i]
                data[cur, i] = lognormal ? x .* ϵ[cur, i] : x .+ ϵ[cur, i]
            else
                x = poly_autoreg(data[prev, i], coefs[i])
                if add_error
                    data[cur, i] = lognormal ? x .* ϵ[cur, i] : x .+ ϵ[cur, i]
                else
                    data[cur, i] = x
                end
            end
        end
    end

    data = size(data, 2) == 1 ? data[:, 1] : data

    return (data = data, epsilon = ϵ)
end

function init_df(N::Int, T::Int, params::AbstractParameters, indp_inputs::AbstractMatrix{<:Real} = Matrix{Float64}(undef, N*T, 0), omega::Vector{<:Real} = Vector{Float64}(undef, 0), Xomegas::AbstractMatrix{<:Real} = Matrix{Float64}(undef, N*T, 0), fixcost::Vector{<:Real} = Vector{Real}(undef, 0), fixcosterr::Bool = false)
    n, n_indp = getnuminputs(params)

    byvars = ["t", "n"]
    fcvar = "FC"
    xvars = genxvars(n)
    ωvars = ["OMEGA$x" for x in [""; "_".*xvars]]
    xivars = ["XI$x" for x in [""; "_".*xvars]]

    ξ = Vector{Union{Missing, Float64}}(missing, N*T)
    Xomegasξ = Matrix{Union{Missing, Float64}}(missing, N*T, size(Xomegas, 2))

    by = [repeat(1:T, inner = N) repeat(1:N, outer = (T))]   
    df = DataFrame(by, byvars) 

    df[!, fcvar] = isempty(fixcost) ? sim_data(N, T, data_params = params.fixcost_params, lognormal = true, add_error = fixcosterr).data : fixcost

    if isempty(omega)
        simomega = sim_data(N, T, data_params = params.omega_params)
        omega = simomega.data 
        ξ = simomega.epsilon
    end
    df[!, ωvars[1]] = omega
    df[!, xivars[1]] = ξ

    if length(Xomegas) < n
        simXomegas = sim_data(N, T, data_params = params.omega_inputs_params)
        Xomegas = [Xomegas simXomegas.data]
        Xomegasξ = [Xomegasξ simXomegas.epsilon]
        if length(Xomegas) < n
            errs = n - length(Xomegas)
            simXomegas = sim_data(N, T; data_params = Params(err = MvNormal(zeros(errs), I)))
            Xomegas = [Xomegas simXomegas.data]
            Xomegasξ = [Xomegasξ simXomegas.epsilon]
        end
    end
    for i in 1:n
        df[!, ωvars[i+1]] = Xomegas[:, i]
        df[!, xivars[i+1]] = Xomegasξ[:, i]
    end

    if size(indp_inputs, 2) < n_indp
        indp_inputs = [indp_inputs sim_data(N, T, data_params = params.indp_params, lognormal = true).data]
        if size(indp_inputs, 2) < n_indp
            errs = n_indp - length(indp_inputs)
            indp_inputs = [indp_inputs sim_data(N, T, data_params = Params(err = MvLogNormal(zeros(errs), I)), lognormal = true).data]
        end
    end
    for i in 1:n_indp
        df[!, xvars[i]] = indp_inputs[:, i]
    end

    return df
end

function solve_firm_decision(df::DataFrame, params::AbstractParameters, closed_solve::Bool = true, xstart::Real = 1000, opt_error::RealOrVec = 0)
    funcs = Functional_Forms()
    prod = params.prod_params
    cost = params.cost_params
    n, n_indp = getnuminputs(params)

    opt_error = [[]; opt_error]

    F_prod = funcs.prod[prod.func]
    F_cost = funcs.cost[cost.func](cost)

    xvars = genxvars(n)

    X_ω = Matrix{Float64}(df[:, names(df, r"OMEGA_")])

    if closed_solve
        X = Matrix{Float64}(df[:, xvars[1:n_indp]])
        sol = F_prod(params, df.OMEGA, X, X_ω)
    else
        sol = solver(df, getnuminputs(params)..., F_prod(prod), F_cost, xstart)
    end

    X = sol.X

    sim_data_validity_check(X, X_ω, df.OMEGA, params, derivative_free = closed_solve)

    for i in (n_indp+1):n
        opt_err_index = i - n_indp
        err = opt_err_index > length(opt_error) ? 1 : rand(LogNormal(0, opt_error[opt_err_index]), nrow(df))
        X[:, i] = X[:, i].*err
        df[!, xvars[i]] = X[:, i]
    end

    df[!, "Y"] = F_prod(prod, nrow(df))(df.OMEGA, X_ω, X) .* exp.(rand(prod.err, nrow(df)))
    df[!, "TC"] = F_cost(X) .+ df.FC
    df[!, "P"] = df.Y - df.TC
    df[!, "termination"] = sol.termination

    for i in 1:n
        input = zeros(nrow(df), n)
        input[:, i] = X[:, i]
        C = F_cost(input)
        df[!, "C_$(xvars[i])"] = C
        df[!, "rent_$(xvars[i])"] = C ./ X[:, i]
        df[!, "shareTC_$(xvars[i])"] = C ./ df.TC
        df[!, "shareY_$(xvars[i])"] = C ./ df.Y
    end
    
    return df
end
