### Objective Function ###
# GMM IV - objective function
# TODO df to include constant, PHI, PHI_lag, X, and X_lag

function ACF(β, g_β, data, vars, g_B, use_constant, closedsolve; derivative = []) 
    # ωstr, ξstr = "omega", "xi"
    constant, PHI, PHI_lag, X, X_lag = data
    NT = length(constant)

    # df[!, ωstr] = PHI - X*β

    OMEGA = PHI - X*β
    OMEGA_lag = PHI_lag - X_lag*β
    # OMEGA = PHI - sum(X[:, i]*β[i] for i in eachindex(β))
    # OMEGA_lag = PHI_lag - sum(X_lag[:, i]*β[i] for i in eachindex(β))


    # df = lag_panel(df, vars.by, [ωstr])
    # df = poly_approx_vars(df, g_B.deg, [ωstr*"_lag"], exponents = g_B.e)
    # df = df[completecases(df), :]

    # X = Matrix(df[:,vars.xvars2])

    # OMEGA_lag_pol = [OMEGA_lag.^i for i in 1:g_B.deg] # Matrix(df[:, names(df, Regex(ωstr*"_lag"))])

    # OMEGA_lag_pol = Matrix(undef, length(constant), g_B.deg + 1)
    # for i in 1:(g_B.deg + 1)
    #     if i == 1
    #         OMEGA_lag_pol[:, i] = constant 
    #     else
    #         OMEGA_lag_pol[:, i] = i == 2 ? OMEGA_lag : OMEGA_lag.^(i-1)
    #     end
    # end

    OMEGA_lag_pol = [constant hcat([OMEGA_lag.^(i) for i in 1:g_B.deg]...)]

    if closedsolve
        # A = Matrix{T}(undef, g_B.deg+1, g_B.deg+1)
        # # # b = Vector(undef, g_B.deg+1)
        # # # mul!(A, OMEGA_lag_pol', OMEGA_lag_pol)
        # # # mul!(b, OMEGA_lag_pol', OMEGA)
        # for i in 1:size(A, 1)
        #     A[:, i] = OMEGA_lag_pol' * OMEGA_lag_pol[:, i]
        # end
        A = OMEGA_lag_pol' * OMEGA_lag_pol
        b = OMEGA_lag_pol' * OMEGA 
    
        # @show A
        # @show b
        dN = 1:NT
        if use_constant
            if isempty(g_B)
                dgbs = 1:length(b)
                gbs = A \ (b)
            else
                gbs = values(g_B, true)
                R = OMEGA .- OMEGA_lag_pol[:, 2:end]*gbs[2:end]
                gbs = [sum(R)/sum(constant); gbs[2:end]]
            end
            # gbs = [sum(OMEGA_lag_pol[j, i]* OMEGA_lag_pol[j, k] for j in dN) for i in dgbs, k in dgbs] \ [sum(OMEGA_lag_pol[j, i] * OMEGA[j] for j in dN) for i in dgbs]
        else
            if isempty(g_B)
                dgbs = 2:length(b)
                A = A[2:end, 2:end]
                b = b[2:end]
                gbs = A \ (b)
                # gbs = [sum(OMEGA_lag_pol[j, i]* OMEGA_lag_pol[j, k] for j in dN) for i in dgbs, k in dgbs] \ [sum(OMEGA_lag_pol[j, i] * OMEGA[j] for j in dN) for i in dgbs]
                gbs = [0; gbs]
            else
                gbs= values(g_B, true)
            end
        end
        return gbs
    end

    if isempty(derivative)
        XI = OMEGA - OMEGA_lag_pol*g_β
        # XI = OMEGA - sum(OMEGA_lag_pol[:, i]*g_β[i] for i in eachindex(g_β))
        # df[!, ξstr] = XI
    else
        d = derivative      
        ngβ = length(g_β)
        if isone(length(derivative))
            if d[1] <= length(β)
                nterms = (ngβ-1)
                if 0 < nterms  
                    # g_prime = OMEGA_lag_pol[:, 1:nterms] * (collect(1:nterms) .* g_β[2:ngβ])
                    g_prime = sum(OMEGA_lag_pol[:, i] * (i * g_β[i+1]) for i in 1:nterms)
                else 
                    g_prime = zeros(Real, NT)
                end
                XI = X[:, d[1]] - X_lag[:, d[1]].*g_prime
            else
                XI = - OMEGA_lag_pol[:, d[1]-length(β)]
            end
        else
            if all(d .<= length(β))
                nterms = (ngβ-2)
                if 0 < nterms  
                    # g_prime = OMEGA_lag_pol[:, 1:nterms] * (collect(1:nterms) .* collect(2:nterms+1) .* g_β[3:ngβ])
                    g_prime = sum(OMEGA_lag_pol[:, i] * (i * (i+1) * g_β[i+2]) for i in 1:nterms)
                else 
                    g_prime = zeros(NT)
                end
                XI = @. - g_prime * X_lag[:, d[1]] * X_lag[:, d[2]]
            elseif all(length(β) .< d) || all((d.-1) .<= length(β))
                XI = zeros(NT)
            else
                dt = d[1] <= length(β) ? d : [d[2], d[1]]
                XI = @. - X_lag[:, dt[1]] * ((dt[2] - length(β) - 1) * OMEGA_lag_pol[:, (dt[2]-length(β)-1)])       
            end
        end
    end
    
    return XI
end

ACF(β, g_β, df::DataFrame, vars, g_B; derivative = []) = ACF(β, g_β, genmodeldf(df, vars), vars, g_B, false, false; derivative = derivative)
ACF(β, df::DataFrame, vars, g_B, use_constant) = ACF(β, [], genmodeldf(df, vars), vars, g_B, use_constant, true)

function gACF(β, df::DataFrame, vars, g_B)
    ωstr, ξstr = "omega", "xi"
    df=deepcopy(df)

    PHI = df[:, vars.phivar]
    X = Matrix(df[:,vars.xvars2])
    X_lag = lag_panel(df[:, [vars.by; vars.xvars2]], vars.by, vars.xvars2)[:, vars.xvars2.*"_lag"]
    X_lag = Matrix(X_lag[completecases(X_lag), :])
    # df[!, ωstr] = PHI - X*β
    df[!, ωstr] = PHI - sum(X[:, i]*β[i] for i in eachindex(β))


    df = lag_panel(df, vars.by, [ωstr])
    df = poly_approx_vars(df, g_B.deg, [ωstr*"_lag"], exponents = g_B.e)
    df = df[completecases(df), :]

    X = Matrix(df[:,vars.xvars2])

    OMEGA_lag_pol = Matrix(df[:, names(df, Regex(ωstr*"_lag"))])

    constant = ones(nrow(df))
    OMEGA_lag_pol = [constant OMEGA_lag_pol]

    dgbs = 2:size(OMEGA_lag_pol,2)
    gbs = [sum(OMEGA_lag_pol[j, i]* OMEGA_lag_pol[j, k] for j in 1:nrow(df)) for i in dgbs, k in dgbs] \ [dot(OMEGA_lag_pol[:, i], df.omega) for i in dgbs]
    gbs = [0; gbs]
   
    return gbs
    
end