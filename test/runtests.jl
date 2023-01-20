using DLWGMMIV
using Test


@testset "DLWGMMIV.jl" begin
    # Write your tests here.
    N,T = 10, 10

    println("Performance of sim_data_CD:")
    @time res1 = DLWGMMIV.sim_data_CD(N,T)

    println("\nPerformance of sim_data_solved_L:")
    @time res2 = DLWGMMIV.sim_data_solved_L(N,T)

    println("\n")
    for res in [["Ipopt",res1], ["Solved L",res2]]
        println("\n\nResults for $(res[1]) data: without ForwardDiff...")
        derivative_checks = DLWGMMIV.sim_data_validity_check_KL(res[2].df, res[2].params)
        @test all(derivative_checks.foc_pass)
        @test all(derivative_checks.soc_pass)

        println("\nResults for $(res[1]) data: with ForwardDiff...")
        derivative_checks = DLWGMMIV.sim_data_validity_check_KL(res[2].df, res[2].params, use_FDiff = true)
        @test all(derivative_checks.foc_pass)
        @test all(derivative_checks.soc_pass)
    end

    df = res2.df
    testdata = [df.time, df.firm, df.Y, df.K, df.L]
    @time results = [dlwGMMIV(testdata...)]
    @time results = [results; dlwGMMIV(testdata..., opt = "LBFGS")]
    @time results = [results; dlwGMMIV(testdata..., prodF ="tl")]
    @time results = [results; dlwGMMIV(testdata..., opt = "LBFGS", prodF = "tl")]

    for res in results
        println("convergence = $(res.conv_msg) |\n  valstart, valend = $(res.valstart), $(res.valend) |\n  betas = $(res.beta_dlw) |\n  g_b = $(res.other_results.g_b)")
    end

end
