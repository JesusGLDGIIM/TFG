# examples/example.jl
using GeneticAlgorithms
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.DifferentialEvolution

# Definir una función objetivo para la prueba
function sphere(x)
    return sum(x .^ 2)
end

# Definir límites de búsqueda
dim = 10
lower_bounds = -5.0
upper_bounds = 5.0
bounds = [lower_bounds, upper_bounds]

pop_size=100
H=100
max_evals=1000000

# Crear instancia de SHADE y usar la interfaz común
shade = SHADE(bounds, dim, pop_size, H, max_evals)
init(shade, sphere)
update(shade, sphere)

println("Best fitness: ", best_fitness(shade))
println("Best solution: ", best_solution(shade))

println("Tamaño de la poblacion; ", population_size(shade))

#println("Historial de mejor fitness: ", shade.best_fitness_history)
#println("Historial de evaluaciones: ", shade.num_eval_history)

# Gráfica de convergencia (requiere Plots.jl)
using Plots
plot(shade.num_eval_history, shade.best_fitness_history, xlabel="Evaluaciones", ylabel="Mejor Fitness", title="Convergencia de SHADE")