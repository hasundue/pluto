### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 421056b1-5fd8-4a43-b9f9-29ab6655f01e
begin
    import Pkg
    Pkg.activate(mktempdir())

    # Registered packages
    pkgnames = [
        "ModelingToolkit",
        "OrdinaryDiffEq",
        "SpecialFunctions",
        "Plots",
        "Printf",
        "RecipesBase",
        "Romberg",
    ]
    pkgspecs = [Pkg.PackageSpec(name = pkgname) for pkgname in pkgnames]
    Pkg.add(pkgspecs)

    # Unregistered packages
    Pkg.add(Pkg.PackageSpec(url="https://github.com/SciML/MethodOfLines.jl"))

    using ModelingToolkit, SpecialFunctions, MethodOfLines, OrdinaryDiffEq
    using Plots, RecipesBase, Printf, Romberg
end

# ╔═╡ aef53a73-6a16-4259-bb35-1d9d735313a0
md"# Mathematical modeling of reactive melt infiltration (RMI)"

# ╔═╡ 8626400d-31e7-4b19-af63-47115eb893c7
md"## Setting up Julia language"

# ╔═╡ bf11b1e1-0459-4c9e-b2a2-cf325f6b7552
md"## Definition of variables and parameters"

# ╔═╡ 51457f69-ed29-4969-bb80-c531ccc7c695
md"#### Variables"

# ╔═╡ 43cd59d9-ad46-42d7-a7ba-443a2bd4f8de
@parameters t x

# ╔═╡ ff609196-76d6-4aba-9971-81835f800271
# c: vapor concentration
# s: surface coverage
# u: liquid volume fraction
# d: thickness of the product at liquid/substrate interface
@variables c(..) s(..) u(..) d(..) T(..)

# ╔═╡ 90323a52-6ec4-45bd-9ee0-11efbab19746
md"#### Model parameters"

# ╔═╡ 8c562f00-b437-4455-b3c2-a04dacb4e77f
# δ: thickness of VW-like crystals at vapor/substrate interface
# r: crystal growth rate
# v: evaporation rate
# D_gas: vapor diffusion coeffient
# w: wettability parameter
# k: permeability parameter
# D_fm: Reaction constant for FM-like reaction at liquid/substrate interface
@parameters δ, r, v, D_gas, w, k, D_fm

# ╔═╡ 6e282286-5366-469d-9f5f-96b85ab2f629
md"#### Physical constants"

# ╔═╡ 5b76a22f-2cd8-4402-b134-9349bd02b232
λ_s = 0.1 # solid thermal conductivity

# ╔═╡ 2ac5c5f2-9882-420a-94f3-17cc9614bf53
λ_l = λ_s # liquid thermal conductivity

# ╔═╡ e8aca2ec-14e4-4dcd-83e1-de898a312eb4
H = 0.1 # enthalpy of reaction

# ╔═╡ 8a2af572-4a87-4f30-a031-56efbda43a70
md"## Governing equations"

# ╔═╡ 3b75ae00-df1a-497b-8553-cec55597fb46
Dx = Differential(x)

# ╔═╡ 547313be-fc25-4868-9331-428976ae14ad
Dxx = Dx^2

# ╔═╡ ede6012f-9cbd-4dbc-b10e-c55b14336c38
Dt = Differential(t)

# ╔═╡ 84ff092f-bde9-4696-91c9-5a5b97a20cde
ev(c, u) = v * (1-c) * u # evaporation model

# ╔═╡ f9d3a857-5fca-4329-ad6e-3a0ab4a53331
vw(c, s) = r * c * (1-s) # VW-like crystal growth at vapor/substrate interface

# ╔═╡ 471e3c14-33ae-4103-8586-41385073f230
cp(w, s) = 1 + erf((s-1)/w) # capillary pressure model

# ╔═╡ 9486d7f2-aa1a-4bdf-828d-6c4dbd1744c3
fm(u, d) = u * D_fm / d # FM-like crystal growth at liquid/substrate interface

# ╔═╡ 8a706e21-9a86-45f6-b329-92f3439d44b1
λ(u) = (1-u) * λ_s + u * λ_l # thermal conductivity

# ╔═╡ ca6af1e9-2bf5-4751-aa95-8b685f4bb987
eqs = [
    # Gas concentration
    Dt(c(t,x)) ~ D_gas * Dxx(c(t,x)) - δ * vw(c(t,x), s(t,x)) + ev(c(t,x), u(t,x)),

    # Surface coverage
    Dt(s(t,x)) ~ vw(c(t,x), s(t,x)),

    # Liquid volume fraction
    Dt(u(t,x)) ~ Dx(k*cp(w, s(t,x))*Dx(u(t,x))),

    # Thickness of the product at liquid/substrate interface
    Dt(d(t,x)) ~ fm(u(t,x), d(t,x)),

    # Temperature
    Dt(T(t,x)) ~ Dx(λ(u(t,x))*Dx(T(t,x))) + H * fm(u(t,x), d(t,x)),
]

# ╔═╡ 59d5fcdd-28cf-44ce-8bd3-bd790c095852
plot([s -> cp(w, s) for w in [10, 1, 0.1]], 0, 1,
     labels = map(w -> "w = " * w, ["10" "1" "0.1"]),
     xlabel = "Surface coverage",
     ylabel = "capillary pressure")

# ╔═╡ a7599967-4633-4b2e-a355-8fcaa0446580
md"## Boundary conditions"

# ╔═╡ b084b8bd-2fe6-4d38-9090-34613ef41842
bcs = [
    c(0,x) ~ 0.0,
    Dx(c(t,0)) ~ 0.0,
    Dx(c(t,1)) ~ 0.0,

    s(0,x) ~ 0.0,
    Dx(s(t,0)) ~ 0.0,
    Dx(s(t,1)) ~ 0.0,

    u(0,x) ~ 0.0,
    u(t,0) ~ 1.0,
    Dx(u(t,1)) ~ 0.0,

    d(0,x) ~ δ,
    Dx(d(t,0)) ~ 0.0,
    Dx(d(t,1)) ~ 0.0,

    T(0,x) ~ 1,
    Dx(T(t,0)) ~ 0.0,
    Dx(T(t,1)) ~ 0.0,
]

# ╔═╡ c8a0e235-9c6d-4d3d-a6c0-75a96db1139b
md"## Solving PDEs"

# ╔═╡ 25ad6ce3-86fc-478f-b957-66857422e2b9
domains = [t ∈ (0.0, 1.0),
           x ∈ (0.0, 1.0)]

# ╔═╡ 1b5c9ffb-70d5-45d8-9042-4892df89008c
@named pde = PDESystem(expand_derivatives.(eqs), 
                       bcs, 
                       domains, 
                       [t,x], 
                       [c(t,x), s(t,x), u(t,x), d(t,x), T(t,x)],
                       [δ => 0.1,       # thickness of VW-like reaction
                        r => 10,        # crystal growth rate
                        v => 40,        # evaporation rate
                        D_gas => 0.01,  # vapor diffusion coeffient
                        w => 0.3,       # wettability parameter
                        k => 10,        # permeability parameter
                        D_fm => 10]);   # Reaction constant for FM-like reaction

# ╔═╡ 4860b5f5-dcb8-40ca-b378-702bcfb40738
dx = 0.01

# ╔═╡ 3f2f9dba-ce74-460a-ae59-78a331de9424
disc = MOLFiniteDifference([x => dx], t);

# ╔═╡ 967e96b7-b29b-480d-9383-2faf8709b13c
prob = discretize(pde, disc)

# ╔═╡ c62ef778-aa47-4079-a580-e9cb5234fe73
sol = solve(prob, Tsit5());

# ╔═╡ b75aac9a-ede8-44db-980e-df35a48b18a3
md"## Plotting results"

# ╔═╡ 93588680-d150-47d1-8cde-8f2c39441881
@recipe function plot(range::StepRangeLen, sol::ODESolution)
    ts = sol.t
    N = length(ts)
    xs = collect(range[2:end-1])
    M = length(xs)

    layout := @layout [c s
                       u l
                       d T]

    t_out = collect(0:0.2:1)
    ls = zeros(length(t_out))
    k = 1
    for i in 1:N
        if ts[i] ≥ t_out[k]
            @series begin
                subplot := 1
                ylabel --> "c"
                xlabel --> "x"
                ylims --> (0, 1)
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][1:M]
            end
            @series begin
                subplot := 2
                ylabel --> "s"
                xlabel --> "x"
                ylims --> (0, 1)
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][M+1:2M]
            end
            @series begin
                subplot := 3
                ylabel --> "u"
                xlabel --> "x"
                ylims --> (0, 1)
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][2M+1:3M]
            end
            @series begin
                subplot := 5
                ylabel --> "d"
                xlabel --> "x"
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][3M+1:4M]
            end
            @series begin
                subplot := 6
                ylabel --> "T"
                xlabel --> "x"
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][4M+1:5M]
            end

            # Calculate infiltration length
            ls[k], _ = romberg(range[2:end-1], sol.u[i][2M+1:3M])

            k += 1
        end
    end
    @series begin
        subplot := 4
        ylabel --> "Infiltration length"
        xlabel --> "t"
        legend --> false
        markershape --> :circle
        t_out, ls
    end
    xlims --> (0, 1)
end

# ╔═╡ eac0b5d5-866d-4a59-b33a-e81eb5c11a4e
plot(0:dx:1, sol, size=(600,800))

# ╔═╡ Cell order:
# ╟─aef53a73-6a16-4259-bb35-1d9d735313a0
# ╟─8626400d-31e7-4b19-af63-47115eb893c7
# ╠═421056b1-5fd8-4a43-b9f9-29ab6655f01e
# ╟─bf11b1e1-0459-4c9e-b2a2-cf325f6b7552
# ╟─51457f69-ed29-4969-bb80-c531ccc7c695
# ╠═43cd59d9-ad46-42d7-a7ba-443a2bd4f8de
# ╠═ff609196-76d6-4aba-9971-81835f800271
# ╟─90323a52-6ec4-45bd-9ee0-11efbab19746
# ╠═8c562f00-b437-4455-b3c2-a04dacb4e77f
# ╟─6e282286-5366-469d-9f5f-96b85ab2f629
# ╠═5b76a22f-2cd8-4402-b134-9349bd02b232
# ╠═2ac5c5f2-9882-420a-94f3-17cc9614bf53
# ╠═e8aca2ec-14e4-4dcd-83e1-de898a312eb4
# ╟─8a2af572-4a87-4f30-a031-56efbda43a70
# ╠═3b75ae00-df1a-497b-8553-cec55597fb46
# ╠═547313be-fc25-4868-9331-428976ae14ad
# ╠═ede6012f-9cbd-4dbc-b10e-c55b14336c38
# ╠═ca6af1e9-2bf5-4751-aa95-8b685f4bb987
# ╠═84ff092f-bde9-4696-91c9-5a5b97a20cde
# ╠═f9d3a857-5fca-4329-ad6e-3a0ab4a53331
# ╠═471e3c14-33ae-4103-8586-41385073f230
# ╠═9486d7f2-aa1a-4bdf-828d-6c4dbd1744c3
# ╠═8a706e21-9a86-45f6-b329-92f3439d44b1
# ╠═59d5fcdd-28cf-44ce-8bd3-bd790c095852
# ╟─a7599967-4633-4b2e-a355-8fcaa0446580
# ╠═b084b8bd-2fe6-4d38-9090-34613ef41842
# ╟─c8a0e235-9c6d-4d3d-a6c0-75a96db1139b
# ╠═25ad6ce3-86fc-478f-b957-66857422e2b9
# ╠═1b5c9ffb-70d5-45d8-9042-4892df89008c
# ╠═4860b5f5-dcb8-40ca-b378-702bcfb40738
# ╠═3f2f9dba-ce74-460a-ae59-78a331de9424
# ╠═967e96b7-b29b-480d-9383-2faf8709b13c
# ╠═c62ef778-aa47-4079-a580-e9cb5234fe73
# ╟─b75aac9a-ede8-44db-980e-df35a48b18a3
# ╠═93588680-d150-47d1-8cde-8f2c39441881
# ╠═eac0b5d5-866d-4a59-b33a-e81eb5c11a4e
