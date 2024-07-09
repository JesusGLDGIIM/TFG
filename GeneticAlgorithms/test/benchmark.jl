module Benchmark
using MAT

export BenchmarkState, benchmark_func, initialize_state

const eval_count = Ref(0)

struct BenchmarkState
    xopt::Vector{Float64}
    p::Vector{Int}
    s::Vector{Int}
    R25::Matrix{Float64}
    R50::Matrix{Float64}
    R100::Matrix{Float64}
    lb::Float64
    ub::Float64
    w::Vector{Float64}
    c::Vector{Int}
    m::Int
end

# Funciones base
function sphere(x)
    return sum(x .^ 2)
end

function elliptic(x)
    D = length(x)
    condition = 1e6
    coefficients = condition .^ range(0, stop=1, length=D)
    return sum(coefficients .* (T_irreg(x) .^ 2))
end

function rastrigin(x)
    D = length(x)
    A = 10
    x = T_diag(T_asy(T_irreg(x), 0.2), 10)
    return A * (D - sum(cos.(2 * π * x))) + sum(x .^ 2)
end

function ackley(x)
    D = length(x)
    x = T_irreg(x)
    x = T_asy(x, 0.2)
    x = T_diag(x, 10)
    return 20 - 20 * exp(-0.2 * sqrt(sum(x .^ 2) / D)) - exp(sum(cos.(2 * π * x)) / D) + exp(1)
end

function schwefel(x)
    D = length(x)
    x = T_asy(T_irreg(x), 0.2)
    fit = 0
    for i in 1:D
        fit += sum(x[1:i])^2
    end
    return fit
end

function rosenbrock(x)
    D = length(x)
    return sum(100 * (x[1:D-1] .^ 2 .- x[2:D]) .^ 2 + (x[1:D-1] .- 1) .^ 2)
end

# Funciones transformadoras
function T_asy(f, beta)
    D = length(f)
    g = copy(f)
    temp = beta * range(0, stop=1, length=D)
    ind = f .> 0
    g[ind] .= f[ind] .^ (1 .+ temp[ind] .* sqrt.(f[ind]))
    return g
end

function T_diag(f, alpha)
    D = length(f)
    scales = sqrt(alpha) .^ range(0, stop=1, length=D)
    return scales .* f
end

function T_irreg(f)
    a = 0.1
    g = copy(f)
    idx = f .> 0
    g[idx] .= log.(f[idx]) / a
    g[idx] .= exp.(g[idx] .+ 0.49 * (sin.(g[idx]) .+ sin.(0.79 .* g[idx]))) .^ a
    idx = f .< 0
    g[idx] .= log.(-f[idx]) / a
    g[idx] .= -exp.(g[idx] .+ 0.49 * (sin.(0.55 .* g[idx]) .+ sin.(0.31 .* g[idx]))) .^ a
    return g
end

function checkBounds(x::Vector{Float64}, lb::Float64, ub::Float64)::Bool
    threshold = 0.001 # Necesario por la aproximacion que hace lbfgsb
    for (i, xi) in enumerate(x)
        if xi < (lb-threshold) || xi > (ub + threshold)
            println("Out of bounds: x[$i] = $xi, lb = $lb, ub = $ub")
            return false
        end
    end
    return true
end

# Funciones de benchmark de f1 a f15
function f1(x, state::BenchmarkState)
    if !checkBounds(x, state.lb, state.ub)
        error("Bounds check failed")
    end
    eval_count[] += 1
    if eval_count[] % 25000 == 0
         println("EvalCount: ", eval_count[])
    end
    x = copy(x .- state.xopt)
    fit = elliptic(x)
    return fit
end

function f2(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = rastrigin(x)
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f3(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = ackley(x)
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f4(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = elliptic(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = elliptic(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = elliptic(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    fit += elliptic(x[state.p[ldim:end]])
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f5(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = rastrigin(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = rastrigin(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = rastrigin(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    fit += rastrigin(x[state.p[ldim:end]])
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f6(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = ackley(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = ackley(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = ackley(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    fit += ackley(x[state.p[ldim:end]])
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f7(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = schwefel(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = schwefel(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = schwefel(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    fit += sphere(x[state.p[ldim:end]])
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f8(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = elliptic(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = elliptic(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = elliptic(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f9(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = rastrigin(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = rastrigin(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = rastrigin(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f10(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = ackley(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = ackley(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = ackley(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f11(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    ldim = 1
    for i in 1:length(state.s)
        if state.s[i] == 25
            f = schwefel(state.R25 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 50
            f = schwefel(state.R50 * x[state.p[ldim:ldim+state.s[i]-1]])
        elseif state.s[i] == 100
            f = schwefel(state.R100 * x[state.p[ldim:ldim+state.s[i]-1]])
        end
        fit += state.w[i] * f
        ldim += state.s[i]
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f12(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = rosenbrock(x)
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f13(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = 0
    for i in 1:length(state.s)
        if i == 1
            ldim = 1
        else
            ldim = state.c[i-1] - ((i-1) * state.m) + 1
        end
        udim = state.c[i] - ((i-1) * state.m)
        if state.s[i] == 25
            f = schwefel(state.R25 * x[state.p[ldim:udim]])
        elseif state.s[i] == 50
            f = schwefel(state.R50 * x[state.p[ldim:udim]])
        elseif state.s[i] == 100
            f = schwefel(state.R100 * x[state.p[ldim:udim]])
        end
        fit += state.w[i] * f
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f14(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    fit = 0
    for i in 1:length(state.s)
        if i == 1
            ldim = 1
            ldimshift = 1
        else
            ldim = state.c[i-1] - ((i-1) * state.m) + 1
            ldimshift = state.c[i-1] + 1
        end
        udim = state.c[i] - ((i-1) * state.m)
        udimshift = state.c[i]
        z = x[state.p[ldim:udim]] .- state.xopt[ldimshift:udimshift]
        if state.s[i] == 25
            f = schwefel(state.R25 * z)
        elseif state.s[i] == 50
            f = schwefel(state.R50 * z)
        elseif state.s[i] == 100
            f = schwefel(state.R100 * z)
        end
        fit += state.w[i] * f
    end
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

function f15(x, state::BenchmarkState)
    idx = checkBounds(x, state.lb, state.ub)
    x = x .- state.xopt
    fit = schwefel(x)
    if !isempty(idx)
        fit[idx] = NaN
    end
    return fit
end

# Función para seleccionar y evaluar la función objetivo correspondiente
function benchmark_func(x, func_num, state::BenchmarkState)
    if func_num == 1
        return f1(x, state)
    elseif func_num == 2
        return f2(x, state)
    elseif func_num == 3
        return f3(x, state)
    elseif func_num == 4
        return f4(x, state)
    elseif func_num == 5
        return f5(x, state)
    elseif func_num == 6
        return f6(x, state)
    elseif func_num == 7
        return f7(x, state)
    elseif func_num == 8
        return f8(x, state)
    elseif func_num == 9
        return f9(x, state)
    elseif func_num == 10
        return f10(x, state)
    elseif func_num == 11
        return f11(x, state)
    elseif func_num == 12
        return f12(x, state)
    elseif func_num == 13
        return f13(x, state)
    elseif func_num == 14
        return f14(x, state)
    elseif func_num == 15
        return f15(x, state)
    else
        error("Function number out of range")
    end
end

function initialize_state(func_num::Int)
    file_num = lpad(func_num, 2, '0')  # Asegura que el número tenga dos dígitos
    file_path = joinpath("test/matlab", "f$file_num.mat")
    
    if isfile(file_path)
        matfile = matread(file_path)
        
        # Extract necessary variables from the .mat file
        xopt = copy(matfile["xopt"][:])
        lb = matfile["lb"]
        ub = matfile["ub"]
        
        # Some functions may not have all these variables, so check their existence
        p = haskey(matfile, "p") ? vec(matfile["p"]) : Vector{Int}()
        s = haskey(matfile, "s") ? vec(matfile["s"]) : Vector{Int}()
        R25 = haskey(matfile, "R25") ? matfile["R25"] : Matrix{Float64}(undef, 0, 0)
        R50 = haskey(matfile, "R50") ? matfile["R50"] : Matrix{Float64}(undef, 0, 0)
        R100 = haskey(matfile, "R100") ? matfile["R100"] : Matrix{Float64}(undef, 0, 0)
        w = haskey(matfile, "w") ? vec(matfile["w"]) : Vector{Float64}()
        m = haskey(matfile, "m") ? Int(matfile["m"]) : 0
        c = haskey(matfile, "c") ? vec(matfile["c"]) : Vector{Int}()
        
        return BenchmarkState(xopt, p, s, R25, R50, R100, lb, ub, w, c, m)
    else
        error("File not found: $file_path")
    end
end

end # module Benchmark