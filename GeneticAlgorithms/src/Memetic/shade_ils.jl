using Random
using Statistics
using Printf
using DelimitedFiles
using GeneticAlgorithms.DifferentialEvolution
using GeneticAlgorithms.MyLocalSearch
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.Utils

function checkBounds(x::Vector{Float64}, lb::Float64, ub::Float64)::Bool
    for (i, xi) in enumerate(x)
        if xi < lb || xi > ub
            println("Out of bounds: x[$i] = $xi, lb = $lb, ub = $ub")
            return false
        end
    end
    return true
end

# Definir la estructura EAresult
struct EAresult
    solution::Vector{Float64}
    fitness::Float64
    evaluations::Int
end

# Implementar PoolLast en Julia
mutable struct PoolLast
    options::Vector{String}
    improvements::Dict{String, Float64}
    count_calls::Int
    first::Vector{String}
    new::Union{Nothing, String}

    function PoolLast(options::Vector{String})
        size = length(options)
        @assert size > 0
        return new(copy(options), Dict(options .=> 0.0), 0, shuffle(copy(options)), nothing)
    end
end



function reset!(pool::PoolLast)
    pool.first = shuffle(copy(pool.options))
    pool.new = nothing
    pool.improvements = Dict(pool.options .=> 0.0)
end

function has_no_improvement(pool::PoolLast)::Bool
    return all(v -> v == 0, values(pool.improvements))
end

function get_new(pool::PoolLast)::String
    if !isempty(pool.first)
        return popfirst!(pool.first)
    end

    if pool.new === nothing
        pool.new = update_prob(pool)
    end

    return pool.new
end

function improvement!(pool::PoolLast, obj::String, account::Float64)
    if account < 0
        return
    end

    if !haskey(pool.improvements, obj)
        error("Error, object not found in PoolLast")
    end

    previous = pool.improvements[obj]
    pool.improvements[obj] = account
    pool.count_calls += 1

    if isempty(pool.first)
        if pool.new === nothing || account == 0 || account < previous
            pool.new = update_prob(pool)
        end
    end
end

function update_prob(pool::PoolLast)::String
    if has_no_improvement(pool)
        return pool.options[rand(1:length(pool.options))]
    end

    improvements_values = collect(values(pool.improvements))
    indexes = sortperm(improvements_values)
    posbest = last(indexes)
    best = collect(keys(pool.improvements))[posbest]
    return best
end


# Función para calcular la mejora
function get_improvement(alg_name, before, after)
    ratio = before == 0 ? 0 : (before - after) / before
    return @sprintf("%s: %.3e -> %.3e [%.2e, %.2f]\n", alg_name, before, after, before - after, ratio)
end

# Función para aplicar búsquedas locales
function apply_localsearch(name, method, fitness_fun, bounds, current_best, current_best_fitness, maxevals, state)
    global SR_MTS
    global SR_global_MTS

    lower = bounds[1]
    upper = bounds[2]

    if method == "grad"
        sol, fit, funcalls = lbfgsb(fitness_fun, current_best, lower, upper, maxevals, state)
        # funcalls = maxevals
    elseif method == "mts"
        SR = name == "global" ? SR_global_MTS : SR_MTS
        res, SR_MTS = mtsls(fitness_fun, current_best, current_best_fitness, lower, upper, maxevals, SR, state)
        sol = res.solution
        fit = res.fitness
        funcalls = maxevals
    else
        error("Method not implemented")
    end

    if fit <= current_best_fitness
        println(get_improvement("$method $name", current_best_fitness, fit))
        return EAresult(sol, fit, funcalls)
    else
        return EAresult(current_best, current_best_fitness, funcalls)
    end
end

# Define a function to calculate the ratio of improvement
function get_ratio_improvement(previous_fitness, new_fitness)
    if previous_fitness == 0
        improvement = 0.0
    else
        improvement = (previous_fitness - new_fitness) / previous_fitness
    end
    return improvement
end

# Variables globales para las búsquedas locales
# global SR_global_MTS
# global SR_MTS

function shadeils(fitness_fun, funinfo, dim, evals, state, popsize = 100, threshold = 0.05, p = 0.1, H = 100)    
    lower = funinfo["lower"]
    upper = funinfo["upper"]
    evals = copy(evals)
    bound = [lower, upper]
    bounds = (fill(lower, dim), fill(upper, dim))
    bounds_partial = (fill(lower, dim), fill(upper, dim))
    popsize = min(popsize, 100)
    maxevals = Int(last(evals))
    totalevals = 0

    # De esto ya se encarga SHADE
    # initial_sol = fill((lower + upper) / 2.0, dim)
    # current_best_fitness = fitness_fun(initial_sol)
    # maxevals = last(evals)
    # totalevals = 1

    
    # population = random_population(lower, upper, dim, popsize)
    # populationFitness = [fitness_fun(ind) for ind in population]
    # bestId = argmin(populationFitness)

    # initial_sol = fill((lower + upper) / 2.0, dim)
    # initial_fitness = fitness_fun(initial_sol)

    # if initial_fitness < populationFitness[bestId]
    #     println("Best initial_sol")
    #     population[bestId] = initial_sol
    #     populationFitness[bestId] = initial_fitness
    # end

    shade = SHADE(bound, dim, popsize, H, maxevals)
    DifferentialEvolution.init(shade, fitness_fun, state, H)
    current_best = EAresult(AbstractAlgorithm.best_solution(shade), AbstractAlgorithm.best_fitness(shade), AbstractAlgorithm.current_evals(shade))
    # crossover = SHADE.SADECrossover(2)
    best_global_solution = current_best.solution
    best_global_fitness = current_best.fitness
    current_best_solution = best_global_solution
    current_best_fitness = best_global_fitness
    
    apply_de = true
    apply_ls = true
    reset_ls(dim, lower, upper)
    methods = ["mts", "grad"]

    pool_global = PoolLast(methods)
    pool = PoolLast(methods)

    num_worse = 0
    evals_gs = min(50 * dim, 25000)
    evals_de = min(50 * dim, 25000)
    evals_ls = min(10 * dim, 5000)
    num_restarts = 0

    totalevals = AbstractAlgorithm.current_evals(shade)

    while totalevals < maxevals
        method = ""

         if !has_no_improvement(pool_global)
            num_best = max(1, round(Int, p * population_size(shade)))
            best_indices = partialsortperm(all_fitness(shade), 1:num_best)
            best_index = rand(best_indices)
            xbest = copy(algo.population[best_index])
            previous_fitness = current_best.fitness
            method_global = get_new(pool_global)
            # println("pool_global")
            # checkBounds(xbest, lower, upper)
            current_best = apply_localsearch("Global", method_global, fitness_fun, bounds, xbest, current_best.fitness, evals_gs, state)
            # println("after pool_global")
            # checkBounds(current_best.solution, lower, upper)
            totalevals += current_best.evaluations
            improvement = get_ratio_improvement(previous_fitness, current_best.fitness)

            shade.population[:,best_index] = copy(current_best.solution)
            shade.fitness[best_index] = current_best[best_index]

            improvement!(pool_global, method_global, improvement)
            evals = check_evals(totalevals, evals, current_best.fitness, best_global_fitness)

            if current_best.fitness < current_best_fitness
                current_best_solution = copy(current_best.solution)
                current_best_fitness = current_best.fitness
            end

            if current_best_fitness < best_global_fitness
                best_global_solution = copy(current_best_solution)
                best_global_fitness = current_best_fitness
            end

            if current_best_fitness < best_fitness(shade)
                shade.best_sol = copy(current_best_solution)
                shade.best_fit = current_best_fitness
            end
        end

        for i in 1:1
            current_best = EAresult(current_best_solution, current_best_fitness, 0)
            set_region_ls()

            method = get_new(pool)

            previous_fitness = current_best.fitness

            if apply_de
                AbstractAlgorithm.update(shade, fitness_fun, state ,evals_de)
                improvement = current_best.fitness - best_fitness(shade)
                totalevals += evals_de
                evals = check_evals(totalevals, evals, best_fitness(shade), best_global_fitness)
                current_best = EAresult(best_solution(shade),best_fitness(shade), totalevals)
                # println("Lower: ", lower, " Upper: ", upper)
                # checkBounds(current_best.solution, lower, upper)
            end

            if apply_ls
                println("before ls $method")
                # eval_count[] = 0
                # checkBounds(current_best.solution, lower, upper)
                result = apply_localsearch("Local", method, fitness_fun, bounds_partial, current_best.solution, current_best.fitness, evals_ls, state)
                # println("after ls $method")
                # checkBounds(result.solution, lower, upper)
                improvement = get_ratio_improvement(current_best_fitness, result.fitness)
                totalevals += result.evaluations
                evals = check_evals(totalevals, evals, result.fitness, best_global_fitness)
                current_best = result

                best_ind = argmin(all_fitness(shade))

                shade.population[:,best_ind] = copy(current_best.solution)
                shade.fitness[best_ind] = current_best.fitness
                
                improvement!(pool, method, improvement)
            end

            current_best_solution = copy(current_best.solution)
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness
                best_global_fitness = current_best_fitness
                best_global_solution = copy(current_best_solution)
            end

            if current_best_fitness < best_fitness(shade)
                shade.best_sol = copy(current_best_solution)
                shade.best_fit = current_best_fitness
            end

            ratio_improvement = previous_fitness == 0 ? 1 : get_ratio_improvement(previous_fitness, current_best.fitness)
            @printf("TotalImprovement[%d%%] %.3e => %.3e (%d)\tRestart: %d\n", (round(100 * ratio_improvement)), previous_fitness, current_best.fitness, num_worse, num_restarts)

            if ratio_improvement >= threshold
                num_worse = 0
            else
                num_worse += 1
                println("Pools Improvements: $(Dict(pool.improvements))")

                reset_ls(dim, lower, upper, method)
            end
            
            # Arreglar el restart
            if num_worse >= 3
                num_worse = 0
                @printf("Restart: %.2e for %.2f with %d evaluations\n", current_best.fitness, ratio_improvement, totalevals)
                posi = rand(1:popsize)
                new_solution = clamp.(rand(dim) * 0.02 .- 0.01 .+ shade.population[:,posi], lower, upper)
                current_best = EAresult(new_solution, fitness_fun(new_solution, state), 0)
                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                reset_de(shade, fitness_fun, state, H, new_solution)
                totalevals += popsize
                reset!(pool_global)
                reset!(pool)
                reset_ls(dim, lower, upper)
                num_restarts += 1
            end

            @printf("%.2e(%.2e): with %d evaluations\n", current_best.fitness, best_global_fitness, totalevals)

            if totalevals >= maxevals
                break
            end
        end
    end

    @printf("%.2e,%s,%d\n", abs(best_global_fitness), join(best_global_solution, " "), totalevals)
    global_res = EAresult(best_global_solution,best_global_fitness,totalevals)
    return global_res, shade
end

function reset_ls(dim, lower, upper, method="all")
    global SR_global_MTS
    global SR_MTS

    if method == "all" || method == "mts"
        SR_global_MTS = fill((upper - lower) * 0.2, dim)
        SR_MTS = copy(SR_global_MTS)
    end
end

function reset_de(algo::SHADE, fun, state, H, p)
    algo.population = Utils.random_population(algo.lower_bounds, algo.upper_bounds,num_dimensions(algo), population_size(algo))
    init(algo, fun, state, H)
    posi = rand(1:population_size(algo))
    algo.population[: ,posi] = p
end

function check_evals(totalevals, evals, bestFitness, globalBestFitness)
    if !isempty(evals) && totalevals >= evals[1]
        best = min(bestFitness, globalBestFitness)
        @printf("[%.1e]: %e,%d\n", evals[1], best, totalevals)
        evals = evals[2:end]
    end
    return evals
end

function set_region_ls()
    global SR_global_MTS
    global SR_MTS

    SR_MTS = copy(SR_global_MTS)
end
#=
# Ejemplo de uso
dim = 10
funinfo = Dict("lower" => -5.0, "upper" => 5.0)
evals = [100, 500, 1000, 50000]
fitness_fun = x -> sum(x .^ 2)
result , shade = shadeils(fitness_fun, funinfo, dim, evals)
println("Best result: ", result)
=#