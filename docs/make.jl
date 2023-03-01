using Documenter
using DLWGMMIV

makedocs(
    sitename = "DLWGMMIV.jl",
    format = Documenter.HTML(),
    modules = [DLWGMMIV],
    pages = [
        "Home" => "index.md",
        ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/kgovernor/DLWGMMIV.jl"
)
