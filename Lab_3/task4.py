import numpy as np
import random


# Knapsack1 – Unbounded version (EA inspired by Lecture 9)

# ----- Fyll i från din uppgift -----
weights = [2, 3, 6, 7, 5]     # example
values  = [6, 5, 8, 9, 6]     # example
capacity = 15                 # example
# ------------------------------------

n = len(weights)

# Max antal av varje item (så sökrymden är begränsad)
max_count = [capacity // w for w in weights]

# 1) Initialization (Population) --------------

def random_individual():
    return [random.randint(0, max_count[i]) for i in range(n)]

def initialize_population(pop_size):
    return [random_individual() for _ in range(pop_size)]

# 2) Fitness function (maximize value)
#    Penalty method for constraint handling (as common in EA)

def fitness(ind):
    total_w = sum(ind[i] * weights[i] for i in range(n))
    total_v = sum(ind[i] * values[i] for i in range(n))

    if total_w <= capacity:
        return total_v
    else:
        # penalty proportional to overweight
        return total_v - 1000 * (total_w - capacity)

# 3) Selection (Tournament selection – mentioned in lecture)
def tournament_selection(pop, k=3):
    contestants = random.sample(pop, k)
    return max(contestants, key=fitness)


# 4) Crossover (Single-point crossover – lecture style)
def crossover(parent1, parent2):
    point = random.randint(1, n-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# 5) Mutation (Gene-wise mutation – lecture mutation idea)
def mutation(ind, mutation_rate=0.1):
    new = ind[:]
    for i in range(n):
        if random.random() < mutation_rate:
            new[i] = random.randint(0, max_count[i])
    return new


# 6) Evolutionary Algorithm (Generational model)
def evolutionary_knapsack(
        pop_size=80,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=2):

    population = initialize_population(pop_size)

    for gen in range(generations):

        # Sort by fitness (elitist replacement)
        population.sort(key=fitness, reverse=True)
        best = population[0]

        new_population = population[:elitism]  # keep best

        while len(new_population) < pop_size:

            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    population.sort(key=fitness, reverse=True)
    best = population[0]

    return best, fitness(best)



# Run/Test -----------------------
if __name__ == "__main__":
    solution, best_fit = evolutionary_knapsack()

    print("Best solution (counts per item):", solution)
    print("Total weight:",
          sum(solution[i]*weights[i] for i in range(n)))
    print("Total value:",
          sum(solution[i]*values[i] for i in range(n)))