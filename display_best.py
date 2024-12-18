import numpy as np
from genomes import Genome, Phenotype, borders_of_continents, display, display_green_blue

arr = np.load("maps/reduced_bool_array.npy").astype(bool)

with open("maps/results_no_antartica/best_genome.txt") as f:
    genome:Genome = eval(f.readlines()[-1])

p = Phenotype(genome, arr.shape)

res = p.classify(arr)
# res[:-1, :-1] ^= borders_of_continents(arr)

# display(res)

cbs = p.circle_borders()
display_green_blue(res[:-1, :-1], np.bitwise_or.reduce(cbs, axis=0))