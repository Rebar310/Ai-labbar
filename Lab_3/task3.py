import numpy as np

# Rook's problem (10 rooks on 10x10) with Evolutionary Algorithm
# Style aligned with lecture: binary representation + crossover + mutation
# inspo from lecture 9

N = 10        # board size
#Defines length of the Chromosome(how many genes it has)
GENE_LEN = N * N  # 100 bits (10x10 flattened)(possible placements)
N_ROOKS = 10    # we want exactly 10 rooks

# One Chromosome will be a binary vector with 100 elements.
# each element will be a place on the board
# row = k // N
# col = k % N

# Makes the chromosome go from 1D-array with 100 genes to 2D-chessboard(phenotype)
def decode(chrom: np.ndarray) -> np.ndarray:
    """Decode 100-bit chromosome into NxN board."""
    return chrom.reshape(N, N) #forms 1x100 to 10x10

# p_one is the probabability that a gene becomes 1
def random_individual(p_one: float = 0.10) -> np.ndarray:
    """
    Create a random chromosome with bits ~ Bernoulli(p_one).
    p_one ~ 0.10 => expected ~10 ones (since 100*0.10=10).
    """
    return (np.random.rand(GENE_LEN) < p_one).astype(int)
    # creates Gene_len (=100) random numbers between 0 and 1
    # then creates an true,false array and converts to (1,0)
    # False = 0, True = 1 
    # Alltså har nu denna skapat en random chromosome = [0,1,0,1,1,0,0,1,0,....]
    # Hopefully the chromosome have near 10 rooks

# decides how good a solution is
# the lower the value the better solution, 0= perfect
def fitness(chrom: np.ndarray,
            w_count: int = 5, # weight for wrong number of rooks
            w_row: int = 10, # weight for row conflict
            w_col: int = 10) -> int: # weight for column conflict
    
    board = decode(chrom) #makes chromosome into chessboard
    rook_count = board.sum() #rooks are represented by ones, therefore we can count

    row_counts = board.sum(axis=1)  # counts rooks/row
    col_counts = board.sum(axis=0)  # counts rooks/col

    # Extra rooks in rows/cols beyond 1 create conflicts.
    row_conflicts = np.sum(np.maximum(0, row_counts - 1)) #sum up how many unneccasary rooks we have
    col_conflicts = np.sum(np.maximum(0, col_counts - 1)) # every value under zero does not count

    # how many rooks is there left on the board, perfekt = abs(10-10) = 0
    count_penalty = abs(rook_count - N_ROOKS)

    # Sum up the fitness value
    return int(w_count * count_penalty + w_row * row_conflicts + w_col * col_conflicts)

# population is a list with all chromosomes and k is how many chromosomes will be tested.
def tournament_select(population, k=3):
    """Pick k random individuals and return the best (lowest fitness)."""
    idx = np.random.choice(len(population), size=k, replace=False) # pick random k index, do not pick the same twice
    candidates = [population[i] for i in idx] # get the chromosomes from the index pick
    return min(candidates, key=fitness) #Run the fitness function and collect the one with best solution


# Matches "Double-point crossover" idea in slides. 
# gets in two parent cromosomes
def double_point_crossover(p1: np.ndarray, p2: np.ndarray):
    a, b = np.sort(np.random.choice(GENE_LEN, size=2, replace=False)) #choose two crossover points, sort them so a is smaller then b
    c1 = p1.copy() #child one copy parent1
    c2 = p2.copy() #cild two copy parent 2
    c1[a:b] = p2[a:b] # just like the slides we now swith the segment 
    c2[a:b] = p1[a:b]
    return c1, c2 # each child will now have been created from parts of their parents


# Matches "Mutation" in slides: flip selected gene(s)
# takes in chromosome and the probability of a gen changeing %
def bitflip_mutation(chrom: np.ndarray, mutation_rate: float = 0.02) -> np.ndarray:
    child = chrom.copy() #make a child identicl to the chromosome
    flip_mask = (np.random.rand(GENE_LEN) < mutation_rate) #create array with random numbers between 0 and 100 and compares with mutation rate
    # flip mask will beccome a true/false array where true means that the gene should mutate
    child[flip_mask] = 1 - child[flip_mask] #this creates the mutation by flipping the values
    return child

# pop_size = how many individuals (chromosomes ) there are in one population
# generations = how many genorations to test on
# crossover rate = probability for crossover
# probability for mutation
# elitsim = how many of the best copies to the next generation
# seed = 0 makes it reproduceable
def solve_rooks_ea(pop_size=80, generations=2000,
                   crossover_rate=0.9, mutation_rate=0.02,
                   elitism=1, seed=0):
    np.random.seed(seed)

    # Initialization
    population = [random_individual(p_one=0.10) for _ in range(pop_size)]
    # creates 80 random chromosomes each 100 bits where 10 bits at least are 1 in averge

    # Loop over generations
    for gen in range(1, generations + 1):
        # Evaluate & sort fitness for one in each population for this generation
        population.sort(key=fitness)
        best = population[0] #the best fit will come first
        best_fit = fitness(best)

        # Termination condition if we find a perfect fit
        if best_fit == 0:
            return best, gen, best_fit

        # Next generation (elitist replacement keeps top 'elitism')
        # copies the best chromosome inte the next generation
        new_pop = population[:elitism]
        
        # Fill upp the new population with children 
        while len(new_pop) < pop_size:
            # Select parents
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            # Crossover 
            if np.random.rand() < crossover_rate: # 90% chance of crossing genes
                c1, c2 = double_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy() #become copies of parents

            # Mutation always happends
            c1 = bitflip_mutation(c1, mutation_rate)
            c2 = bitflip_mutation(c2, mutation_rate)
            
            # Fill up with individuals
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop #This will change to the new generation of children

    # If no perfect solution found, return best found
    population.sort(key=fitness)
    return population[0], generations, fitness(population[0])


# -------------------------------
# 7) Run + present solution
# -------------------------------
if __name__ == "__main__":
    best, gen, best_fit = solve_rooks_ea()

    board = decode(best)
    print(f"Solved? {'YES' if best_fit==0 else 'NO'} | best fitness={best_fit} | generation={gen}")
    print("Rook count:", board.sum())

    # Print board
    # R = rook, . = empty
    for r in range(N):
        row = ["R" if board[r, c] == 1 else "." for c in range(N)]
        print(" ".join(row))

    # Also show row/col counts (helpful debug)
    print("Row counts:", board.sum(axis=1))
    print("Col counts:", board.sum(axis=0))