using DLWGMMIV
using Test


@testset "DLWGMMIV.jl" begin
    # Write your tests here.

    ### Testing Sim data ###
    println("\n\n-----TESTING SIM DATA FUNCTIONS-----\n")
    N,T = 1000, 15
    params = Parameters(
		# omega_coefs = [0, 0.8],
		# cost_exps = [1, 1.15]
	)
    model = ACF_model(
		use_constant = false,
		omega_deg = 3,
		stage2_deg = 2,
	)

    by = ["t", "n"]
    yvar = ["Y"]
    xvars1 = ["x1","x2"]
    xvars2 = ["x1","x2"]
    zvars = ["x1","x2_lag"]

    vars = (
        by = ["t", "n"],
        yvar = ["Y"],
        xvars1 = ["x1","x2"],
        xvars2 = ["x1","x2"],
        zvars = ["x1","x2_lag"],
        phivar = "phi"
    )

    # println("\nPerformance of sim_data_solved_L:")
    # @time res_SL = DLWGMMIV.sim_data_solved_L_CD(N,T)

    println("\nPerformance of sim_data:")
    df = simfirmdata(N, T, params = params, seed = 1)
    
    println("\n==========================================")
    ########################################################

    ### Testing dlwGMMIV function ###
    println("\n\n-----TESTING DLWGMMIV FUNCTION-----\n")
    df[:, unique([yvar; xvars1; xvars2])] = log.(df[:, unique([yvar; xvars1; xvars2])])
    df = lag_panel(df, by, ["x2"])

    res = dlwGMMIV(
        df, by, yvar, xvars1, xvars2, zvars;  
        betas = Betas(),
        bstart = [0.5, 0.5, 0, 0 ,0] ,
        model = model,
        skip_stage1 = true,
        globalsolve = false,
        df_out = true
    );

    @show res.r.s2.beta_dlw

    println("\n==========================================")
    ##############################################################

    ### Testing gnrGMMIV function ###
    println("\n\n-----TESTING GNRGMMIV FUNCTION-----\n")

end
