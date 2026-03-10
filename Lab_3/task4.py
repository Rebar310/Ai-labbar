import numpy as np
import random


# Knapsack1 – Unbounded version (EA inspired by Lecture 9)

# Vikter (kg) för varje item 
weights = [1, 1, 2, 4, 12]

# Värde ($) för varje item 
values = [1, 2, 2, 10, 4]

capacity = 15  # Max vikt ryggsäcken klarar
            
n = len(weights) # Antal olika items

# Max antal av varje item (så sökrymden är begränsad)
max_count = [capacity // w for w in weights] # ex: item med vikt 4kg -> max 3 st (15//4)


# 1) Initialization (Population) --------------

def random_individual():
    return [random.randint(0, max_count[i]) for i in range(n)]
# Skapar en slumpmässig lösning.
# Exempel: [0,2,1,4,3]
# Detta betyder item1 = 0 st, item2 = 2 st


def initialize_population(pop_size):
    return [random_individual() for _ in range(pop_size)]
# Skapar en population av slumpmässiga lösningar


# 2) Fitness function (maximize value)
#    Penalty method for constraint handling (as common in EA)
def fitness(individual):
    
    total_w = sum(individual[i] * weights[i] for i in range(n)) #total vikt
    total_v = sum(individual[i] * values[i] for i in range(n)) #totalt värde

    if total_w <= capacity:
        return total_v
    else:
        # penalty proportional to overweight
        penalty_factor = 1000
        return total_v - penalty_factor * (total_w - capacity)

# 3) Selection (Tournament selection – mentioned in lecture)
def tournament_selection(population, k=3):
    contestants = random.sample(population, k) #väljer tre random individer
    return max(contestants, key=fitness)
#Funktionen används för att välja parents

# 4) Crossover (Single-point crossover – lecture style)
def crossover(parent1, parent2):
    point = random.randint(1, n-1) #väljer slumpmässig punkt i listan
    child1 = parent1[:point] + parent2[point:] #gör så att ena barnet får börsta början av ena föräldern och andra delen av andra föräldern
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# 5) Mutation (Gene-wise mutation – lecture mutation idea)
def mutation(individual, mutation_rate=0.3):
    new = individual[:] #gör kopia av individen
    for i in range(n):
        if random.random() < mutation_rate: #random.random ger ett tal mellan 0 och 1
            new[i] = random.randint(0, max_count[i]) # om mutation sker ändra värdet på den platsen mellan 0-5
    return new


# 6) Evolutionary Algorithm (Generational model)
def evolutionary_knapsack(
        pop_size=80,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.3,
        elitism=1):

    population = initialize_population(pop_size) #skapa population av individuals

    for gen in range(generations):

        # Sort by fitness (elitist replacement)
        population.sort(key=fitness, reverse=True) #bästa lösningen = högst fitness
        best = population[0] #bästa lösningen hamnar först i listan

        new_population = population[:elitism]  # keep best

        while len(new_population) < pop_size:

            # Selection, slumpar ut parents
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

            # worst_fit = fitness(population[-1])

            # c1_fit = fitness(child1)
            # c2_fit = fitness(child2)

            # if c1_fit > worst_fit:
            #     new_population.append(child1)
            # else:
            #     new_population.append(parent1[:])

            # if len(new_population) < pop_size:
            #     if c2_fit > worst_fit:
            #         new_population.append(child2)
            # else:
            #     new_population.append(parent2[:])
             

        population = new_population #nu byts generation alltså ny population

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