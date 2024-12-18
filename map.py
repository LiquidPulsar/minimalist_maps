from random import choices, randint, random
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from genomes import Genome, Phenotype, borders_of_continents, display

arr = np.load("maps/reduced_bool_array.npy").astype(bool)



# def accuracy(genome: Genome, image: np.ndarray) -> float:
#     """
#     F1 score of how well the circles fit the image
#     """
#     f1_score

print(arr.shape, arr.dtype, arr.mean())

p = Phenotype(
    [
        (50, 50, 30),
        (arr.shape[1] // 2, arr.shape[0] // 2, 30),
        (arr.shape[1] // 2 - 30, arr.shape[0] // 2 - 30, 20),
        (arr.shape[1] // 2 - 30, arr.shape[0] // 2 + 10, 20),
    ],
    arr.shape,
)
print(p.masks.shape)

res = p.classify(arr)

# combo = p.get_region(3)

res[:-1, :-1] ^= borders_of_continents(arr)

# display(res)

print(p.score(arr))

##############################################
#           GENETIC ALGORITHM                #
##############################################


def bounded(x: int, low: int, high: int) -> int:
    return max(low, min(x, high))


def mutate(genome: Genome, w: int, h: int, scale: int = 1) -> Genome:
    new_genome = genome.copy()
    i = np.random.randint(len(genome))
    x, y, r = genome[i]
    new_r = max(0, r + np.random.randint(-5, 5) * scale)
    # if your radius is small, you probably don't matter much
    # so we can move you around a bit
    if r < 10:
        scale *= 5
    elif r < 20:
        scale *= 2
    new_genome[i] = (
        bounded(x + np.random.randint(-10, 10) * scale, -new_r, w + new_r),
        bounded(y + np.random.randint(-10, 10) * scale, -new_r, h + new_r),
        new_r,
    )
    return new_genome


def crossover(genome1: Genome, genome2: Genome) -> Genome:
    new_genome = []
    for g1, g2 in zip(genome1, genome2):
        new_genome.append(g1 if np.random.rand() < 0.5 else g2)
    return new_genome


def select(
    population: list[tuple[Genome, float]], n: int
) -> list[tuple[Genome, float]]:
    return sorted(population, key=lambda x: x[1], reverse=True)[:n]


def genetic_algorithm(
    mu: int,
    lambda_: int,
    image: np.ndarray,
    n_generations: int,
    num_circles: int,
    *,
    save=False,
    results_dir="results",
) -> list[Genome]:
    h, w = image.shape
    population = [
        [(randint(0, w), randint(0, h), 30) for _ in range(num_circles)]
        for _ in range(mu + lambda_)
    ]
    for generation in trange(n_generations):
        scored_population = [
            (genome, Phenotype(genome, (h, w)).score(image)) for genome in population
        ]
        parents = select(scored_population, mu)
        unscored_parents = [p for (p, _) in parents]
        tqdm.write(f"{[s for (_, s) in parents]}")
        for i in range(1, len(unscored_parents)):
            if random() < 0.5:
                unscored_parents[i] = mutate(unscored_parents[i], w, h, scale=6)
        children = []
        for _ in range(lambda_):
            parent1, parent2 = choices(unscored_parents, k=2)
            child = mutate(crossover(parent1, parent2), w, h, scale=3)
            children.append(child)
        population = unscored_parents + children

        if save:
            scored_population = [
                (genome, p := Phenotype(genome, arr.shape), p.score(arr))
                for genome in population
            ]
            scored_population.sort(key=lambda x: x[2], reverse=True)
            best = scored_population[0][1].classify(arr)
            best[:-1, :-1] ^= borders_of_continents(arr)
            Image.fromarray(best.astype(np.uint8) * 255, mode="L").save(
                f"maps/{results_dir}/genetic_algorithm_{generation}.png"
            )
            with open(f"maps/{results_dir}/best_genome.txt", "a") as f:
                f.write(f"{scored_population[0][0]}\n")
    return population


arr = arr[:178] # remove antartica


population = genetic_algorithm(10, 5, arr, 10_000, 5, save=False, results_dir="results_no_antartica")
scored_population = [
    (genome, p := Phenotype(genome, arr.shape), p.score(arr)) for genome in population
]
scored_population.sort(key=lambda x: x[2], reverse=True)
print(scored_population[0])
best = scored_population[0][1].classify(arr)
best[:-1, :-1] ^= borders_of_continents(arr)

display(best)
