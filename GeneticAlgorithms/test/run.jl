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
# using GeneticAlgorithms.Benchmark
using GeneticAlgorithms.Memetic

include("./cec2013lsgo.jl")

# Configuración
NP = 100  # Tamaño de la población
runs = 1  # Número de ejecuciones independientes
evals = [Int(1.2e5), Int(6e5), Int(3e6)]  # Máximo número de evaluaciones
maxevals = Int(3e6)
H = 100

global aux = 0

# Definir la función de límites
function get_bounds(func_num)
    if func_num in [1, 4, 7, 8, 11, 12, 13, 14, 15]
        return -100.0, 100.0
    elseif func_num in [2, 5, 9]
        return -5.0, 5.0
    elseif func_num in [3, 6, 10]
        return -32.0, 32.0
    else
        error("Función no reconocida")
    end
end

# Función para evaluar el algoritmo
function evaluate_shadeils(NP, runs, evals, func_num)
    results = Dict{Int, Vector{Float64}}()
    lb, ub = get_bounds(func_num)

    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "test/cec2013lsgo")
    cec2013_set_data_dir("test/cec2013lsgo/cdatafiles")

    for run in 1:runs
        funinfo = Dict("lower" => lb, "upper" => ub)
        
        fitness_fun = (f = x -> cec2013_eval_sol(x))

        result, shade = shadeils(fitness_fun, funinfo, D, evals, NP)
        if !haskey(results, func_num)
            results[func_num] = Float64[]
        end
        push!(results[func_num], result.fitness)
        
        println("Run $run: Best fitness = $(result.fitness)")

        cec2013_next_run()
    end
    
    for func_num in keys(results)
        mean_result = mean(results[func_num])
        std_result = std(results[func_num])
        println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    end

    

    return results
end

function evaluate_shade(NP, runs, maxevals, H, func_num)
    results = Dict{Int, Vector{Float64}}()
    lb, ub = get_bounds(func_num)
    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "test/cec2013lsgo")
    cec2013_set_data_dir("test/cec2013lsgo/cdatafiles")

    for run in 1:runs
        fitness_fun = x -> cec2013_eval_sol(x)
        bound = [lb, ub]
        
        shade = SHADE(bound, D, NP, H, maxevals)
        init(shade, fitness_fun, H)
        update(shade, fitness_fun, maxevals)
        if !haskey(results, func_num)
            results[func_num] = Float64[]
        end
        push!(results[func_num], best_fitness(shade))
        
        println("Run $run: Best fitness = $(best_fitness(shade))")

        cec2013_next_run()

        plot(shade.num_eval_history, shade.best_fitness_history, xlabel="Evaluaciones", ylabel="Mejor Fitness", title="Convergencia de SHADE")
    end

    for func_num in keys(results)
        mean_result = mean(results[func_num])
        std_result = std(results[func_num])
        println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    end

    return results
end

# Ejecutar la evaluación
for i in 1:15
    results_shadeils = evaluate_shadeils(NP, runs, evals, i)
    #results_shade = evaluate_shade(NP, runs, maxevals, H, i)
end
