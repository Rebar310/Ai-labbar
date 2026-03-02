import random
from typing import List, Tuple

# ============================================================
# Rook's problem: 10 rooks on 10x10 board, no shared row/column
# Evolutionary algorithm (Genetic Algorithm)
# ============================================================

N = 10  # board size and number of rooks

# ------------------------------------------------------------
# 1) Representation (Genotype -> Phenotype)
# ------------------------------------------------------------

Genome = List[int]

def random_genome() -> Genome:
    """Create a random permutation (candidate solution)."""
    g = list(range(N))
    random.shuffle(g)
    return g

# ------------------------------------------------------------
# 2) Fitness function
# ------------------------------------------------------------

def rook_conflicts(genome: Genome) -> int:
    """
    Count number of attacking pairs (same row or same column).
    - Row is unique by construction (index), so no row conflicts.
    - Column conflicts occur if genome has duplicates (not with permutations).
    """
    # Count column duplicates:
    cols = genome
    conflicts = 0
    # For each column, count how many rooks are there; if k rooks share it,
    # they create k*(k-1)/2 attacking pairs.
    for c in set(cols):
        k = cols.count(c)
        if k > 1:
            conflicts += k * (k - 1) // 2
    return conflicts


def fitness(genome: Genome) -> int:
    """Lower fitness is better (0 = solved)."""
    return rook_conflicts(genome)

# ------------------------------------------------------------
# 3) Selection (Tournament selection)
# ------------------------------------------------------------
# Select parents biased toward better fitness.
# ------------------------------------------------------------

def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    """Pick k random individuals and return the best among them."""
    contestants = random.sample(population, k)
    return min(contestants, key=fitness)

# ------------------------------------------------------------
# 4) Crossover (Order Crossover - OX) for permutations
# ------------------------------------------------------------
# Must keep permutation property (no duplicates).
# ------------------------------------------------------------

def order_crossover(p1: Genome, p2: Genome) -> Genome:
    """
    Order crossover (OX):
    - Copy a slice from parent1
    - Fill remaining positions in the order they appear in parent2
    """
    a, b = sorted(random.sample(range(N), 2))
    child = [None] * N

    # Copy slice from p1
    child[a:b+1] = p1[a:b+1]

    # Fill remaining from p2 in order, skipping used values
    used = set(child[a:b+1])
    fill_positions = [i for i in range(N) if child[i] is None]
    fill_values = [x for x in p2 if x not in used]

    for pos, val in zip(fill_positions, fill_values):
        child[pos] = val

    # type checker: child is now complete permutation
    return child  # type: ignore

# ------------------------------------------------------------
# 5) Mutation (Swap mutation) for permutations
# ------------------------------------------------------------

def swap_mutation(genome: Genome, p_mut: float = 0.2) -> Genome:
    """With probability p_mut, swap two positions (rows)."""
    g = genome[:]
    if random.random() < p_mut:
        i, j = random.sample(range(N), 2)
        g[i], g[j] = g[j], g[i]
    return g


# ------------------------------------------------------------
# 6) Evolution loop (GA / EA)
# ------------------------------------------------------------

def solve_rooks_ea(
    pop_size: int = 50,
    generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    tournament_k: int = 3,
    elitism: int = 1,
    seed: int | None = 0,
) -> Tuple[Genome, int]:
    if seed is not None:
        random.seed(seed)

    # Initialize population
    population = [random_genome() for _ in range(pop_size)]

    for gen in range(1, generations + 1):
        # Sort by fitness to apply elitism and for reporting
        population.sort(key=fitness)
        best = population[0]
        best_fit = fitness(best)

        # Termination: solved
        if best_fit == 0:
            print(f"Solved at generation {gen}, fitness={best_fit}")
            return best, gen

        # Build next generation
        new_pop: List[Genome] = []

        # Elitism: keep top individuals
        new_pop.extend(population[:elitism])

        while len(new_pop) < pop_size:
            # Selection
            p1 = tournament_select(population, k=tournament_k)
            p2 = tournament_select(population, k=tournament_k)

            # Crossover
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]  # clone

            # Mutation
            child = swap_mutation(child, p_mut=mutation_rate)

            new_pop.append(child)

        population = new_pop

    # If not found early, return best of last generation
    population.sort(key=fitness)
    return population[0], generations


# ------------------------------------------------------------
# 7) Run and present result
# ------------------------------------------------------------

if __name__ == "__main__":
    solution, gens = solve_rooks_ea()

    print("\nFinal solution genome (row -> column):")
    print(solution)
    print("Conflicts:", fitness(solution))

    # Pretty-print as board
    print("\nBoard (R = rook):")
    for r in range(N):
        row = ["." for _ in range(N)]
        row[solution[r]] = "R"
        print(" ".join(row))