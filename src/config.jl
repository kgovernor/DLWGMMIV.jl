######################
# Data Configuration #
######################

# This function configures the data for GMMIV method. Input 1 is treated as the dynamic input. 
function DataConfig(year, plantid, Q, input1, input2, inputs...)
    q, inputs = log.(Q), [log.(inp) for inp in [[input1, input2]; collect(inputs)]]
    components = poly_approx_vars(inputs...)
    df = DataFrame( ; year = year, plantid=plantid, q = q, components...)

    # TO DO - Let phi be derived from polynomial regression of degree 3.
    df[!, :phi] = q  
    df[!, :epsilon] = df.q .- df.phi

    sort!(df, [:plantid, :year])
    gdf = groupby(df, :plantid)

    transform!(gdf, :phi => (x -> lag(x)) => :phi_lag)

    vars = ["x"*string(i) for i in eachindex(inputs)]
    for var in vars
        transform!(gdf, Symbol(var*"1") => (x -> lag(x)) => Symbol(var*"_lag"))
        transform!(gdf, Symbol(var*"2") => (x -> lag(x)) => Symbol(var*"_lag2"))
    end
    
    for var in combinations(vars, 2)
        df[!, var[1]*"_lag"*var[2]*"_lag"] = df[:, var[1]*"_lag"] .* df[:, var[2]*"_lag"]
    end

    for var in vars[2:length(vars)]
        df[!, vars[1]*var*"_lag"] = df[:, vars[1]*"1"] .* df[:, var*"_lag"]
    end

    df[!, :lq_c] = df.q .- df.epsilon
    df[!, :qlvl_c] = exp.(df.lq_c)
    df[!, :constant] = ones(length(df.q))

    df = df[completecases(df), :]
    sort!(df, [:plantid, :year])
    
    return df
end

function poly_approx_vars(input1, input2, inputs...)
    M, N = 3, 3 # degree of polynomial    
    inputs = [[input1, input2]; collect(inputs)] # Production function inputs
    X_str = ["x"*string(i) for i in eachindex(inputs)] # labels for inputs
    X = Dict(zip(X_str,inputs))
    X_cmb = combinations(X_str, 2) # creates set of interactions between inputs.

    components = Dict()
    # Create polynomial terms needed for configuring data for GMMIV
    for i in 1:M
        for x in X_str
            components[x*string(i)] = X[x] .^ (i)
        end


        for j in 1:N
            for x_cmb in X_cmb
                components[join(x_cmb,string(i))*string(j)] = X[x_cmb[1]] .^ (i) .* X[x_cmb[2]] .^ (j)
            end
        end

        if length(X_str) > 2
            components[join(X_str,string(i))*string(i)] = prod.(eachrow(hcat(inputs...).^i))
        end
    end

    components = Dict(Symbol(k) => v for (k, v) in components)

    return components
end
