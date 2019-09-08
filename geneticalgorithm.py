import numpy as np
import geneticfunctions as gp

pop_num = 10                 # size of a population
best_num = 2                # how many of the pop is kept unchanged for the next generation
rest_num = pop_num - best_num    # How many individuals are not kept from the previous generation
equation = [-4, 5, 0, 5, 3, -19, 45]      # what those weights are
weights = len(equation)                  # number of weights in the equation

general_pop = np.random.rand(pop_num, weights)

#print(general_pop)

parent_num = 8
gen_num = 1000


for gen in range(gen_num):
    fitness = gp.fitness(equation, general_pop)
    # Work out fitness of each of individual
    fitness_indices = np.argsort(fitness)[::-1][:2]
    # returns the indices of the individuals in order of fitness
    # reverses the list and gets the first two elements
    # i.e. the ones with the largest fitnesses
    new_pop = []
    # New population
    for i in fitness_indices:
        new_pop.append(general_pop[i])
    # Keep the most successful elements of the last generation in the new one
    parents = gp.nextgen(general_pop, fitness, parent_num)
    # Probabilistically determine next parents
    offspring = gp.crossover(parents, parent_num, rest_num)
    # Uses parents to breed and create offspring
    offspring = gp.mutate(offspring)
    # Offspring are then mutated
    for i in range(rest_num):
        new_pop.append(offspring[i])
    # Adds this offspring to the new population
    general_pop = np.array(new_pop)
    # The new population is now the general population

print("Final result is ")
print(general_pop)
print("Fitnesses are: ")
print(general_pop.dot(equation))