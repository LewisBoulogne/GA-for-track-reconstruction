import matplotlib.pyplot as plt

from random import randint

"""
Simplified case:

2 tracks 3 layers

hits:

h1 - h3 - h5
h2 - h4 - h6

segments:

 . s1 . s5 .
   \s2  \s6
   /s3  /s7
 . s4 . s8 .

individual = binar string [s1, s2, s3, s4, s5, s6, s7, s8]

"""

solution = "100110011001"
all_scores = []
population_size = 10
os_amount = int(population_size/3)
runs = 3

def init():
    tracks = []
    for r in range(population_size):
        ind = ""
        for i in range(len(solution)):
            ind += str(randint(0,1))
        tracks.append(ind)
    return tracks


def found(popu):
    # check whether the tracks correspond to solution
    global solution
    if solution in popu:
        return True
    else:
        return False


def get_fit(population):
    global solution, mean_score
    scores = []
    for ind in population:
        score = 0
        for i in range(len(solution)):
            if ind[i-1] == solution[i-1]:
                score += 1
        scores.append((ind, score))
    sum = 0
    for tuple in scores:
        sum += tuple[1]
    av_score = sum / len(population)
    fittest = []
    for tup in scores:
        if tup[1] >= av_score:
            fittest.append(tup[0])
    if len(fittest) != int(len(population)/2):
        if len(fittest) < int(len(population)/2):
            for r in range(int(len(population)/2) - len(fittest)):
                indiv = ""
                for j in range(len(solution)):
                    indiv += str(randint(0,1))
                fittest.append(indiv)
        if len(fittest) > int(len(population)/2):
            for c in range(len(fittest) - int(len(population)/2)):
                rand_index = randint(0, len(fittest)-1)
                fittest.pop(rand_index)
    mean_score.append(av_score)
    return fittest

def os_pop(parents): #off-springs
    global solution, os_amount
    next_gen = []
    for par in range(os_amount):
        if par%2 == 0:
            p1 = parents[par]
            p2 = parents[par - 1]
            cross_point = randint(0, len(solution) - 1)
            os1 = p1[0: cross_point] + p2[cross_point: len(solution)]
            os2 = p2[0: cross_point] + p1[cross_point: len(solution)]
            next_gen.append(os1)
            next_gen.append(os2)
    return next_gen

def next_pop(population):
    next_pop = []
    fittest = get_fit(population)
    for fit in fittest:
        next_pop.append(fit)
    offspring = os_pop(fittest)
    for os in offspring:
        next_pop.append(os)
    if len(next_pop) != len(population):
        if len(next_pop) < len(population):
            for i in range(len(population) - len(next_pop)):
                ind = ""
                for i in range(len(solution)):
                    ind += str(randint(0,1))
                next_pop.append(ind)
        if len(next_pop) > len(population):
            for j in range(len(next_pop) - len(population)):
                rand_index = randint(0, len(next_pop)-1)
                next_pop.pop(rand_index)
    return next_pop
    

def main_lvl1():
    global mean_score, all_scores, runs
    for r in range(runs):
        mean_score = []
        generation = 1
        pop = init()
        print(pop)
        while not found(pop) and generation < 100:
            print(generation)
            pop = next_pop(pop)
            print(pop)
            generation += 1
        if found(pop):
            print("Solved in " + str(generation) + " generations")
        if generation == 100:
            print("Generation limit reached")
        all_scores.append(mean_score)
    for score_list in all_scores:
        plt.plot(score_list)
    plt.show()
    

main_lvl1()