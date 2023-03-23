using DLWGMMIV
using Test


@testset "DLWGMMIV.jl" begin
    # Write your tests here.

    ### Testing Sim data ###
    println("\n\n-----TESTING SIM DATA FUNCTIONS-----\n")
    N,T = 10, 10

    # println("\nPerformance of sim_data_solved_L:")
    # @time res_SL = DLWGMMIV.sim_data_solved_L_CD(N,T)

    println("\nPerformance of sim_data:")
    res = []
    @time res = [res; DLWGMMIV.sim_data(N,T)]
    @time res = [res; DLWGMMIV.sim_data(N,T, prod_params = [0.1, 0.25, 0.05], prodF ="tl")]
    @time res = [res; DLWGMMIV.sim_data(N,T, num_inputs = 3, input_names = ["k", "l", "m"], prod_params = [0.1, 0.25, 0.6])]
    @time res = [res; DLWGMMIV.sim_data(N,T, num_inputs = 3, input_names = ["k", "l", "m"], prod_params = [0.1, 0.25, 0.2, 0.05], prodF ="tl")]

    println("\n==========================================")
    ########################################################

    ### Testing Solved_L simmed data ###
    # println("\n\n-----TESTING SOLVED_L SIMMED DATA-----\n")
    # println("\n\nResults for solved_L data: without ForwardDiff...")
    # derivative_checks = DLWGMMIV.sim_data_validity_check_solved_L_CD(res_SL.df, res_SL.params)
    # @test all(derivative_checks.foc_pass)
    # @test all(derivative_checks.soc_pass)

    # println("\nResults for solved_L data: with ForwardDiff...")
    # derivative_checks = DLWGMMIV.sim_data_validity_check_solved_L_CD(res_SL.df, res_SL.params, use_FDiff = true)
    # @test all(derivative_checks.foc_pass)
    # @test all(derivative_checks.soc_pass)
    # println("\n==========================================")
    ###############################################################

    ### Testing simmed data ###
    println("\n\n-----TESTING SIMMED DATA-----\n")
    for r in res
        println("### Simmed Data for $(r.input_params.num_inputs) inputs, $(r.params.prodF) ###")
        if (r.input_params.num_inputs == 2) && (r.params.prodF == "CD")
            println("\n\nResults for data: without ForwardDiff...")
            derivative_checks = DLWGMMIV.sim_data_validity_check(r.df, r.params, r.funcs, r.input_params)
            @test all(derivative_checks.foc_pass)
            @test all(derivative_checks.soc_pass)
        end

        println("\nResults for data: with ForwardDiff...")
        derivative_checks = DLWGMMIV.sim_data_validity_check(r.df, r.params, r.funcs, r.input_params, use_FDiff = true)
        @test all(derivative_checks.foc_pass)
        @test all(derivative_checks.soc_pass)
        println("\n\n")
    end
    println("\n==========================================")
    ###############################################################

    ### Testing dlwGMMIV function ###
    println("\n\n-----TESTING DLWGMMIV FUNCTION-----\n")
    opt_str = ["NelderMead", "LBFGS"]
    prodF_str = ["CobbDouglas", "TransLog"]
    description = [[opt, prodF] for prodF in prodF_str for opt in opt_str]

    # # For solved L data
    # println("Results for solved_L data:")
    # df_SL = res_SL.df
    # testdata_SL = [df_SL.time, df_SL.firm, df_SL.Y, df_SL.K, df_SL.L]
    # @time results = [dlwGMMIV(testdata_SL...)]
    # @time results = [results; dlwGMMIV(testdata_SL..., opt = "LBFGS")]
    # @time results = [results; dlwGMMIV(testdata_SL..., prodF ="tl")]
    # @time results = [results; dlwGMMIV(testdata_SL..., opt = "LBFGS", prodF = "tl")]

    # for i in eachindex(results)
    #     res_iv = results[i]
    #     desc = description[i]

    #     println("  For $(desc[1]) Optimization, solve $(desc[2]): ")
    #     println("  \tconvergence = $(res_iv.conv_msg) |\n  \tvalstart, valend = $(res_iv.valstart), $(res_iv.valend) |\n  \tbetas = $(res_iv.beta_dlw) |\n  \tg_b = $(res_iv.other_results.g_b) \n\n")
    # end

    # For simmed data
    for r in res
        println("Results for Simmed Data $(r.input_params.num_inputs) inputs, $(r.params.prodF)\n")
        df = r.df
        testdata = [df.time, df.firm, df.Y, [df[:,input] for input in r.input_params.input_names]...]
        @time results = [dlwGMMIV(testdata..., num_indp_inputs = r.input_params.num_indp_inputs)]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = r.input_params.num_indp_inputs, opt = "LBFGS")]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = r.input_params.num_indp_inputs, prodF ="tl")]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = r.input_params.num_indp_inputs, opt = "LBFGS", prodF = "tl")]

        for i in eachindex(results)
            res_iv = results[i]
            desc = description[i]

            println("  For $(desc[1]) Optimization, solve $(desc[2]): ")
            println("  \tconvergence = $(res_iv.conv_msg) |\n  \tvalstart, valend = $(res_iv.valstart), $(res_iv.valend) |\n  \tbetas = $(res_iv.beta_dlw) |\n  \tg_b = $(res_iv.other_results.g_b) \n\n")
        end
    end

    # Test adding constant to dlwGMMIV
    res_test = DLWGMMIV.sim_data(1000, 10)
    df_test = res_test.df
    testdata = [df_test.time, df_test.firm, df_test.Y, [df_test[:,input] for input in res_test.input_params.input_names]...]
    for use_constant in ["", "X", "Z", "omega", "notZ", "all"]
        println("Results for Simmed Data $(res_test.input_params.num_inputs) inputs, $(res_test.params.prodF), constant in $(use_constant) \n")
        @time results = [dlwGMMIV(testdata..., num_indp_inputs = res_test.input_params.num_indp_inputs, use_constant = use_constant)]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = res_test.input_params.num_indp_inputs, opt = "LBFGS", use_constant = use_constant)]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = res_test.input_params.num_indp_inputs, prodF ="tl", use_constant = use_constant)]
        @time results = [results; dlwGMMIV(testdata..., num_indp_inputs = res_test.input_params.num_indp_inputs, opt = "LBFGS", prodF = "tl", use_constant = use_constant)]

        for i in eachindex(results)
            res_iv = results[i]
            desc = description[i]

            println("  For $(desc[1]) Optimization, solve $(desc[2]): ")
            println("  \tconvergence = $(res_iv.conv_msg) |\n  \tvalstart, valend = $(res_iv.valstart), $(res_iv.valend) |\n  \tbetas = $(res_iv.beta_dlw) |\n  \tg_b = $(res_iv.other_results.g_b) \n\n")
        end
    end

    println("\n==========================================")
    ##############################################################

end
