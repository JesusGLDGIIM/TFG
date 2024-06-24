using Distributions

# Definir la función objetivo (para minimizar)
function objective_function(x)
    return sum(x .^ 2)
end

# Inicializar la población
function initialize_population(pop_size, dim, bounds)
    population = [rand(bounds[1]:bounds[2], dim) for _ in 1:pop_size]
    return population
end

# Generar vector de prueba
function generate_trial_vector(target, pbest, rand1, rand2, F, CR, dim)
    v = target + F * (pbest - target) + F * (rand1 - rand2)
    u = similar(target)
    j_rand = rand(1:dim)
    for j in 1:dim
        if rand() < CR || j == j_rand
            u[j] = v[j]
        else
            u[j] = target[j]
        end
    end
    return u
end

# Actualizar memoria
function update_memory(M, success_params)
    if length(success_params) > 0
        mean_param = mean(success_params)
        M[end] = mean_param
        M = circshift(M, 1)
    end
    return M
end

# Algoritmo SHADE
function SHADE(pop_size, dim, bounds, max_gen, H)
    # Inicializar población y parámetros
    population = initialize_population(pop_size, dim, bounds)
    fitness = [objective_function(ind) for ind in population]
    
    M_F = fill(0.5, H)
    M_CR = fill(0.5, H)
    A = []
    k = 1
    g = 0
    
    while g < max_gen
        S_F, S_CR = Float64[], Float64[]
        
        for i in 1:pop_size
            r = rand(1:H)
            F = clamp(rand(Normal(M_F[r], 0.1)), 0.1, 1.0)
            CR = clamp(rand(Normal(M_CR[r], 0.1)), 0.0, 1.0)
            p = rand(0.1:0.1:0.2)
            p_best = sortperm(fitness)[1:round(Int, p * pop_size)]
            best = population[rand(p_best)]
            
            candidates = setdiff(1:pop_size, i)
            r1, r2 = randperm(candidates, 2)
            
            trial = generate_trial_vector(population[i], best, population[r1], population[r2], F, CR, dim)
            trial_fitness = objective_function(trial)
            
            if trial_fitness <= fitness[i]
                population[i] = trial
                fitness[i] = trial_fitness
            end
            if trial_fitness < fitness[i]
                push!(A, population[i])
                push!(S_F, F)
                push!(S_CR, CR)
            end
        end
        
        if length(A) > pop_size
            A = A[randperm(end)[1:pop_size]]
        end
        
        if !isempty(S_F) && !isempty(S_CR)
            M_F[k] = mean(S_F)
            M_CR[k] = mean(S_CR)
            k += 1
            if k > H
                k = 1
            end
        end
        
        g += 1
    end
    
    best_index = argmin(fitness)
    return population[best_index], fitness[best_index]
end

# Ejemplo de uso
pop_size = 100
dim = 10
bounds = (-5.0, 5.0)
max_gen = 1000
H = 10

best_solution, best_fitness = SHADE(pop_size, dim, bounds, max_gen, H)
println("Best solution: ", best_solution)
println("Best fitness: ", best_fitness)
