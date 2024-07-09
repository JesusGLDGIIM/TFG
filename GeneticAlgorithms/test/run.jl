using Printf
using Random
using Statistics
using MAT
using Plots
using GeneticAlgorithms
using GeneticAlgorithms.DifferentialEvolution
using GeneticAlgorithms.MyLocalSearch
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.Utils
using GeneticAlgorithms.Benchmark
using GeneticAlgorithms.Memetic

# Configuración
NP = 100  # Tamaño de la población
runs = 1  # Número de ejecuciones independientes
evals = [1.2e5, 6e5, 3e6]  # Máximo número de evaluaciones
maxevals = Int(3e6)
H = 100

# Función para evaluar el algoritmo
function evaluate_shadeils(NP, runs, evals)
    results = Dict{Int, Vector{Float64}}()

    for func_num in 1:1
        state = initialize_state(func_num)
        D = length(state.xopt)

        funinfo = Dict("lower" => state.lb[1], "upper" => state.ub[1])
        
        fitness_fun = (x, state) -> benchmark_func(x, func_num, state)
        
        for run in 1:runs
            result, shade = shadeils(fitness_fun, funinfo, D, evals, state, NP)
            if !haskey(results, func_num)
                results[func_num] = Float64[]
            end
            push!(results[func_num], result.fitness)
            
            println("Run $run: Best fitness = $(result.fitness)")
        end
    end

    for func_num in keys(results)
        mean_result = mean(results[func_num])
        std_result = std(results[func_num])
        println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    end

    return results
end

function evaluate_shade(NP, runs, maxevals, H)
    results = Dict{Int, Vector{Float64}}()

    for func_num in 1:1
        state = initialize_state(func_num)
        D = length(state.xopt)
        
        fitness_fun = (x, state) -> benchmark_func(x, func_num, state)

        bound = [state.lb, state.ub]
        
        for run in 1:runs
            shade = SHADE(bound, D, NP, H, maxevals)
            init(shade, fitness_fun, state, H)
            update(shade, fitness_fun, state, maxevals)
            if !haskey(results, func_num)
                results[func_num] = Float64[]
            end
            push!(results[func_num], best_fitness(shade))
            
            println("Run $run: Best fitness = $(best_fitness(shade))")

            plot(shade.num_eval_history, shade.best_fitness_history, xlabel="Evaluaciones", ylabel="Mejor Fitness", title="Convergencia de SHADE")
        end
    end

    for func_num in keys(results)
        mean_result = mean(results[func_num])
        std_result = std(results[func_num])
        println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    end

    return results
end


# Ejecutar la evaluación
# results = evaluate_shade(NP, runs, maxevals, H)
results = evaluate_shadeils(NP, runs, evals)
println(eval_count[])
