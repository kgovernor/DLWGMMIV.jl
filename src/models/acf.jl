### Objective Function ###
# GMM IV - objective function
# TODO rewrite function so that it is more easily acceptable to different forms of beta. Expand matrix algebra notation

function ACF(β, g_β, df, vars, g_B, use_constant, closedsolve; derivative = [])
    ωstr, ξstr = "omega", "xi"

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

    if closedsolve
        if use_constant
            dgbs = 1:size(OMEGA_lag_pol,2)
            # gbs = OMEGA_lag_pol'OMEGA_lag_pol \ OMEGA_lag_pol'df.omega
            gbs = [sum(OMEGA_lag_pol[j, i]* OMEGA_lag_pol[j, k] for j in 1:nrow(df)) for i in dgbs, k in dgbs] \ [dot(OMEGA_lag_pol[:, i], df.omega) for i in dgbs]
        else
            dgbs = 2:size(OMEGA_lag_pol,2)
            # gbs = OMEGA_lag_pol[:,2:end]' * OMEGA_lag_pol[:,2:end] \ OMEGA_lag_pol[:,2:end]' * df.omega
            gbs = [sum(OMEGA_lag_pol[j, i]* OMEGA_lag_pol[j, k] for j in 1:nrow(df)) for i in dgbs, k in dgbs] \ [dot(OMEGA_lag_pol[:, i], df.omega) for i in dgbs]
            gbs = [0; gbs]
        end
        return gbs
    end

    if isempty(derivative)
        # XI = df[:, ωstr] - OMEGA_lag_pol*g_β
        XI = df[:, ωstr] - sum(OMEGA_lag_pol[:, i]*g_β[i] for i in eachindex(g_β))
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
                    g_prime = zeros(Real,nrow(df))
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
                    g_prime = zeros(nrow(df))
                end
                XI = @. - g_prime * X_lag[:, d[1]] * X_lag[:, d[2]]
            elseif all(length(β) .< d) || all((d.-1) .<= length(β))
                XI = zeros(nrow(df))
            else
                dt = d[1] <= length(β) ? d : [d[2], d[1]]
                XI = @. - X_lag[:, dt[1]] * ((dt[2] - length(β) - 1) * OMEGA_lag_pol[:, (dt[2]-length(β)-1)])       
            end
        end
    end
    
    return XI
end

ACF(β, g_β, df::DataFrame, vars, g_B; derivative = []) = ACF(β, g_β, df, vars, g_B, false, false; derivative = derivative)
ACF(β, df::DataFrame, vars, g_B, use_constant) = ACF(β, [], df, vars, g_B, use_constant, true)

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