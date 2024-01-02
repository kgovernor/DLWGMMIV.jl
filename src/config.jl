######################
# Data Configuration #
######################

function lag_panel(df::DataFrame, by::Vector{String}, vars::Vector{String} = Vector{String}(undef, 0))
    if isempty(vars)
        vars = names(df, Not(by))
    end

    sort!(df, by)
    gdf = groupby(df, by[2])

    df = transform(gdf, [x => ShiftedArrays.lag for x in vars])

    return df
end

function poly_vars(vars::Vector{String}, deg::Int; exponents::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, 0))
    poly_vars = Vector{String}(undef, 0)
    
    if isempty(exponents)
        if deg > 1
            for d in 2:deg
                append!(exponents, collect(multiexponents(length(vars), d)))
            end
        end
    end
    
    for e in exponents
        sel = e .> 0
        e = replace(e, 1=>"")
        push!(poly_vars,  join(vars[sel] .* string.(e[sel])))
    end

    return (v = poly_vars, e = exponents)
end

function poly_approx_vars(df::DataFrame, deg::Int, vars::Vector{String} = Vector{String}(undef, 0); exponents::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, 0))
    vars = isempty(vars) ? names(df) : vars
    pv, exponents = poly_vars(vars, deg, exponents = exponents)

    for e in exponents
        newvar = popfirst!(pv)
        if !(newvar in names(df))
            df = transform(df, [vars[i] .=> (x -> x.^e[i]) for i in eachindex(vars)] )
            df = select(df, names(df, Not(r"function"))..., names(df, r"function")  => ByRow(*) => newvar)
        end
    end

    return df
end
