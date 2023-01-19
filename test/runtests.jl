using DLWGMMIV
using Test


@testset "DLWGMMIV.jl" begin
    # Write your tests here.
    println("Performance of sim_data_CD:")
    @time res1 = sim_data_CD(10,10)
    first(res1.df, 10)

    println("\nPerformance of sim_data_solved_L:")
    @time res2 = sim_data_solved_L(10,10)
    first(res2.df, 10)

    println("\n")
    for res in [["Ipopt",res1], ["Solved L",res2]]
        println("\n\nResults for $(res[1]) data: without ForwardDiff...")
        derivative_checks = DLWGMMIV.sim_data_validity_check_KL(res[2][1], res[2][2])
        @test all(derivative_checks.foc_pass)
        @test all(derivative_checks.soc_pass)

        println("\nResults for $(res[1]) data: with ForwardDiff...")
        derivative_checks = DLWGMMIV.sim_data_validity_check_KL(res[2][1], res[2][2], use_FDiff = true)
        @test all(derivative_checks.foc_pass)
        @test all(derivative_checks.soc_pass)
    end
end
