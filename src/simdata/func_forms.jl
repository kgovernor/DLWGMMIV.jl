function check_func(funcs, func)
    if !(func in funcs)
        throw(ArgumentError("$func invalid functional form"))
    end
end

function CobbDouglas(prod_params::Params, N::Int = 1)
    p = prod_params.i
    coefs = values(prod_params, p.coefs)
    n = length(coefs)
    c = ones(N,n)
    
    function f(ω::RealOrVec, X_ω::AbstractArray{<:Real}, X::AbstractArray{<:Real}; derivative::Vector{Int} = Vector{Int}(undef,0))
        d = derivative
        X_ω = X_ω isa Vector ? X_ω' : X_ω
        X = X isa Vector ? X' : X
        while !isempty(d)
            i = popfirst!(d)
            ci = coefs[i]
            c[:, i] = @. c[:, i] * ci * exp(X_ω[:, i])
            if !iszero(ci)
                coefs[i] -= 1
            end
        end
        v = exp.(ω)
        for i in eachindex(coefs)
            v = @. v * c[:, i] * ((exp(X_ω[:, i]) * X[:, i])^coefs[i])
        end

        coefs = values(prod_params, p.coefs)
        c = ones(N, n)
        v = isone(length(v)) ? v[1] : v
        return v
    end
    return f
end

function CobbDouglas(params::AbstractParameters, ω::RealOrVec, X::AbstractMatrix{<:Real}, X_ω::AbstractMatrix{<:Real})
    p = params.prod_params.i
    n, n_indp = getnuminputs(params)
    indps = 1:n_indp
    deps = (n_indp+1):n
    X_opt = Matrix(undef, length(ω), n)
    if !isempty(X)
        X_opt[:, indps] = X[:, indps]
    end

    p_coefs = values(params.prod_params, p.coefs)
    c_coefs = values(params.cost_params, p.coefs)
    c_exps = values(params.cost_params, p.exps)

    γi = 1 ./ (c_exps[deps])
    αi = p_coefs[deps] ./ γi
    A = exp.(ω)
    for i in eachindex(p_coefs)
        @. A *= exp(X_ω[:, i])^p_coefs[i]
    end 

    A_bar = prod((αi ./ c_coefs[deps]) .^ (αi))
    for i in indps
        A_bar = @. A_bar * (X[:, i]^p_coefs[i])
    end

    α_bar = sum(αi)

    Y_opt = @. (A * A_bar) ^ (1/(1-α_bar))
    for d in deps
        X_opt[:, d] = @. ((αi[d-n_indp] / c_coefs[d]) * Y_opt) ^ (γi[d-n_indp])
    end
    termination = repeat(["closed_solve"], length(Y_opt))

    return FirmSolution(Y_opt, X_opt, termination) 
end

function CES()
end

function VES()
end

function Translog()
end

function AddSeperable(cost_params::Params)
    p = cost_params.i
    coefs = values(cost_params, p.coefs)
    exps = values(cost_params, p.exps)
    c = ones(length(coefs))

    function f(X::AbstractArray{<:Real}; derivative::Vector{Int} = Vector{Int}(undef,0))
        d = derivative
        X = X isa Vector ? X' : X
        v = zeros(Real, size(X, 1))

        while !isempty(d) 
            i = popfirst!(d)
            ei = exps[i]
            for j in eachindex(c)
                c[j] = j == i ? c[j]*ei : 0 
            end
            if !iszero(ei)
                exps[i] -= 1
            end
        end
        for i in eachindex(coefs)
            @. v += c[i] * coefs[i] * (X[:, i]^(exps[i]))
        end

        coefs = values(cost_params, p.coefs)
        exps = values(cost_params, p.exps)
        c = ones(length(coefs))

        v = isone(length(v)) ? v[1] : v
        return v
    end

    return f
end
