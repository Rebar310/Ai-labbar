import numpy as np

# This task has much inspiration from Lecture 9
# ============================================================
# Rook's problem (10 rooks on 10x10) with Evolutionary Algorithm
# Style aligned with lecture: binary representation + crossover + mutation
# ============================================================

N = 10                 # board size
GENE_LEN = N * N       # 100 bits (10x10 flattened)
N_ROOKS = 10           # we want exactly 10 rooks

# -------------------------------
# 1) Representation (Genotype)
# -------------------------------
# Chromosome = binary string / vector of length 100:
#   gene[k] in {0,1}
# Mapping gene index -> board cell:
#   row = k // N, col = k % N
# gene[k]=1 means: place a rook at (row, col).
# Phenotype = the decoded 10x10 board.
# -------------------------------

def decode(chrom: np.ndarray) -> np.ndarray:
    """Decode 100-bit chromosome into NxN board."""
    return chrom.reshape(N, N)

def random_individual(p_one: float = 0.10) -> np.ndarray:
    """
    Create a random chromosome with bits ~ Bernoulli(p_one).
    p_one ~ 0.10 => expected ~10 ones (since 100*0.10=10).
    """
    return (np.random.rand(GENE_LEN) < p_one).astype(int)

# -------------------------------
# 2) Fitness function (minimize)
# -------------------------------
# We want:
# - exactly 10 rooks
# - <=1 rook per row
# - <=1 rook per column
#
# Penalty-based fitness (lower is better, 0 is perfect):
#   penalty = w_count * |#rooks - 10|
#           + w_row   * sum_row max(0, row_count-1)
#           + w_col   * sum_col max(0, col_count-1)
# -------------------------------

def fitness(chrom: np.ndarray,
            w_count: int = 5,
            w_row: int = 10,
            w_col: int = 10) -> int:
    board = decode(chrom)
    rook_count = board.sum()

    row_counts = board.sum(axis=1)  # length 10
    col_counts = board.sum(axis=0)  # length 10

    # Extra rooks in rows/cols beyond 1 create conflicts.
    row_conflicts = np.sum(np.maximum(0, row_counts - 1))
    col_conflicts = np.sum(np.maximum(0, col_counts - 1))

    count_penalty = abs(rook_count - N_ROOKS)

    return int(w_count * count_penalty + w_row * row_conflicts + w_col * col_conflicts)

# -------------------------------
# 3) Selection (Tournament)
# -------------------------------
def tournament_select(population, k=3):
    """Pick k random individuals and return the best (lowest fitness)."""
    idx = np.random.choice(len(population), size=k, replace=False)
    candidates = [population[i] for i in idx]
    return min(candidates, key=fitness)

# -------------------------------
# 4) Crossover (Double-point)
# -------------------------------
# Matches "Double-point crossover" idea in slides.
def double_point_crossover(p1: np.ndarray, p2: np.ndarray):
    a, b = np.sort(np.random.choice(GENE_LEN, size=2, replace=False))
    c1 = p1.copy()
    c2 = p2.copy()
    c1[a:b] = p2[a:b]
    c2[a:b] = p1[a:b]
    return c1, c2

# -------------------------------
# 5) Mutation (Bit-flip)
# -------------------------------
# Matches "Mutation" in slides: flip selected gene(s)
def bitflip_mutation(chrom: np.ndarray, mutation_rate: float = 0.02) -> np.ndarray:
    child = chrom.copy()
    flip_mask = (np.random.rand(GENE_LEN) < mutation_rate)
    child[flip_mask] = 1 - child[flip_mask]
    return child

# -------------------------------
# 6) EA main loop (Formulation)
# -------------------------------
# Initialization -> Population
# repeat:
#   Select parents
#   Crossover
#   Mutation
#   Evaluate
#   Elitist replacement (keep best)
# until termination condition (fitness==0 or max generations)
# -------------------------------

def solve_rooks_ea(pop_size=80, generations=2000,
                   crossover_rate=0.9, mutation_rate=0.02,
                   elitism=1, seed=0):
    np.random.seed(seed)

    # Initialization
    population = [random_individual(p_one=0.10) for _ in range(pop_size)]

    for gen in range(1, generations + 1):
        # Evaluate & sort
        population.sort(key=fitness)
        best = population[0]
        best_fit = fitness(best)

        # Termination condition
        if best_fit == 0:
            return best, gen, best_fit

        # Next generation (elitist replacement keeps top 'elitism')
        new_pop = population[:elitism]

        while len(new_pop) < pop_size:
            # Select
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            # Crossover
            if np.random.rand() < crossover_rate:
                c1, c2 = double_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bitflip_mutation(c1, mutation_rate)
            c2 = bitflip_mutation(c2, mutation_rate)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

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