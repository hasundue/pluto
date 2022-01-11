### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 421056b1-5fd8-4a43-b9f9-29ab6655f01e
begin
    import Pkg
    Pkg.activate(mktempdir())

    # Registered packages
    pkgnames = ["ModelingToolkit",
                "OrdinaryDiffEq",
                "SpecialFunctions",
                "Plots",
                "Printf",
                "RecipesBase"]
    pkgspecs = [Pkg.PackageSpec(name = pkgname) for pkgname in pkgnames]
    Pkg.add(pkgspecs)

    # Unregistered packages
    Pkg.add(Pkg.PackageSpec(url="https://github.com/SciML/MethodOfLines.jl"))

    using ModelingToolkit, SpecialFunctions, MethodOfLines, OrdinaryDiffEq
    using Plots, RecipesBase, Printf
end

# ╔═╡ 43cd59d9-ad46-42d7-a7ba-443a2bd4f8de
@parameters t x

# ╔═╡ ff609196-76d6-4aba-9971-81835f800271
@variables c(..) s(..) u(..)

# ╔═╡ 3b75ae00-df1a-497b-8553-cec55597fb46
Dx = Differential(x)

# ╔═╡ 547313be-fc25-4868-9331-428976ae14ad
Dxx = Dx^2

# ╔═╡ ede6012f-9cbd-4dbc-b10e-c55b14336c38
Dt = Differential(t)

# ╔═╡ 8c562f00-b437-4455-b3c2-a04dacb4e77f
δ = 0.1 # thickness parameter

# ╔═╡ 2bc4b9de-c5e1-4bdb-832c-f6c849b0dc3d
r = 10 # crystal growth rate

# ╔═╡ 3e67f206-6c54-4406-b431-8e0dc70cc862
v = 10 # evaporation rate

# ╔═╡ 7af0447f-aca2-45b0-ac1c-5d359ec25b3a
D = 0.1 # gas diffusion coeffient

# ╔═╡ 85271f34-f54c-4f94-be61-46db68d4be3c
w = 0.3 # wettability parameter

# ╔═╡ 56c4d260-3d8d-4c83-8f5a-283878c76d9b
k = 10 # infiltrativity parameter

# ╔═╡ 84ff092f-bde9-4696-91c9-5a5b97a20cde
ev(c, u) = v * (1-c) * u # evaporation

# ╔═╡ f9d3a857-5fca-4329-ad6e-3a0ab4a53331
cg(c, s) = r * c^2 * (1-s)^2 # crystal growth

# ╔═╡ 471e3c14-33ae-4103-8586-41385073f230
cp(s) = 1 + erf((s-1)/w) # relative capillary pressure

# ╔═╡ ca6af1e9-2bf5-4751-aa95-8b685f4bb987
eqs = [
    Dt(c(t,x)) ~ D * Dxx(c(t,x)) - δ * cg(c(t,x), s(t,x)) + ev(c(t,x), u(t,x)),
    Dt(s(t,x)) ~ cg(c(t,x), s(t,x)),
    Dt(u(t,x)) ~ Dx(k*cp(s(t,x))*Dx(u(t,x))),
]

# ╔═╡ b084b8bd-2fe6-4d38-9090-34613ef41842
bcs = [
    c(t,0) ~ 1.0,
    c(0,x) ~ 0.0,
    s(t,0) ~ 1.0,
    s(0,x) ~ 0.0,
    u(t,0) ~ 1.0,
    u(0,x) ~ 0.0,
]

# ╔═╡ 25ad6ce3-86fc-478f-b957-66857422e2b9
domains = [t ∈ (0.0, 1.0),
           x ∈ (0.0, 1.0)]

# ╔═╡ 1b5c9ffb-70d5-45d8-9042-4892df89008c
@named pde = PDESystem(expand_derivatives.(eqs), bcs, domains, [t,x], [c(t,x), s(t,x), u(t,x)]);

# ╔═╡ 4860b5f5-dcb8-40ca-b378-702bcfb40738
dx = 0.01

# ╔═╡ 3f2f9dba-ce74-460a-ae59-78a331de9424
disc = MOLFiniteDifference([x => dx], t);

# ╔═╡ 967e96b7-b29b-480d-9383-2faf8709b13c
prob = discretize(pde, disc)

# ╔═╡ c62ef778-aa47-4079-a580-e9cb5234fe73
sol = solve(prob, Tsit5());

# ╔═╡ 93588680-d150-47d1-8cde-8f2c39441881
@recipe function plot(xs::StepRangeLen, sol::ODESolution)
    ts = sol.t
    N = length(ts)
    xs = collect(xs[2:end])
    M = length(xs)

    layout := @layout [c s
                       u u]

    tout = collect(0:0.2:1)
    for i in 1:N
        if ts[i] ≥ tout[1]
            tout = tout[2:end]
            @series begin
                subplot := 1
                ylabel --> "gas"
                xlabel --> "x"
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][1:M]
            end
            @series begin
                subplot := 2
                ylabel --> "coverage"
                xlabel --> "x"
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][M+1:2M]
            end
            @series begin
                subplot := 3
                ylabel --> "liquid"
                xlabel --> "x"
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][2M+1:3M]
            end
        end
    end
    xlims --> (0, 1)
    ylims --> (0, 1)
end

# ╔═╡ eac0b5d5-866d-4a59-b33a-e81eb5c11a4e
plot(0:dx:1, sol)

# ╔═╡ Cell order:
# ╠═421056b1-5fd8-4a43-b9f9-29ab6655f01e
# ╠═43cd59d9-ad46-42d7-a7ba-443a2bd4f8de
# ╠═ff609196-76d6-4aba-9971-81835f800271
# ╠═3b75ae00-df1a-497b-8553-cec55597fb46
# ╠═547313be-fc25-4868-9331-428976ae14ad
# ╠═ede6012f-9cbd-4dbc-b10e-c55b14336c38
# ╠═8c562f00-b437-4455-b3c2-a04dacb4e77f
# ╠═2bc4b9de-c5e1-4bdb-832c-f6c849b0dc3d
# ╠═3e67f206-6c54-4406-b431-8e0dc70cc862
# ╠═7af0447f-aca2-45b0-ac1c-5d359ec25b3a
# ╠═85271f34-f54c-4f94-be61-46db68d4be3c
# ╠═56c4d260-3d8d-4c83-8f5a-283878c76d9b
# ╠═ca6af1e9-2bf5-4751-aa95-8b685f4bb987
# ╠═84ff092f-bde9-4696-91c9-5a5b97a20cde
# ╠═f9d3a857-5fca-4329-ad6e-3a0ab4a53331
# ╠═471e3c14-33ae-4103-8586-41385073f230
# ╠═b084b8bd-2fe6-4d38-9090-34613ef41842
# ╠═25ad6ce3-86fc-478f-b957-66857422e2b9
# ╠═1b5c9ffb-70d5-45d8-9042-4892df89008c
# ╠═4860b5f5-dcb8-40ca-b378-702bcfb40738
# ╠═3f2f9dba-ce74-460a-ae59-78a331de9424
# ╠═967e96b7-b29b-480d-9383-2faf8709b13c
# ╠═c62ef778-aa47-4079-a580-e9cb5234fe73
# ╠═93588680-d150-47d1-8cde-8f2c39441881
# ╠═eac0b5d5-866d-4a59-b33a-e81eb5c11a4e
