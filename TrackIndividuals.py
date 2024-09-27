from random import randint
import matplotlib.pyplot as plt
import numpy as np

"""
Simplified case:

2 tracks 5 layers

hits:

h1 - h3 - h5 - h7 - h9
h2 - h4 - h6 - h8 - h10

segments:

 . s1 . s5 .
  's2  's6
  ,s3  ,s7
 . s4 . s8 .

individual = int string 5 long e.g. [h1, h3, h5, h7, h9]

hit = (x, y) e.g h1 = (0, 1), h2 = (0, 0)

"""

tracks = 2
layers = 5
population_size = 20
max_gen = 1000
os_amount = int(population_size/4)
fit_size = int(population_size/2)
max_pop_score = [0]
mean_pop_score =[0]
best_track = []


def init_pop(size):
    pop = []
    for i in range(size):
        ind = []
        for layer in range(layers):
            ind.append((layer, randint(0, tracks - 1)))
        pop.append(ind)
    return pop


def get_angle(a, b, c):
    #AB.BC = len(AB).len(BC).cos(angle)
    AB_v = (abs(a[0]-b[0]), abs(a[1]-b[1]))
    BC_v = (abs(b[0]-c[0]), abs(b[1]-c[1]))
    AB_dot_BC = np.dot(AB_v, BC_v)#AB[0]*BC[0] + AB[1]*BC[1]
    len_AB = np.sqrt((AB_v[0]^2) + (AB_v[1]^2))
    len_BC = np.sqrt((BC_v[0]^2) + (BC_v[1]^2))
    if (len_AB * len_BC) == 0:
        return 1
    else:
        return AB_dot_BC / (len_AB * len_BC)


def angle_score(track):
    a_score = 0
    for hit in range(len(track)-1):
        if hit != 0: 
            if get_angle(track[hit - 1], track[hit], track[hit + 1]) == 0:
                a_score += 1
    return a_score


def get_fit(population):
    global best_track
    scores = []
    for ind1 in population:
        ind_score = angle_score(ind1)
        scores.append(ind_score)
        if ind_score > max(max_pop_score):
            best_track = ind1
    av_score = np.mean(scores)
    fittest = []
    for ind2 in population:
        if angle_score(ind2) >= av_score:
            fittest.append(ind2)
    fittest.sort()
    if len(fittest) > fit_size:
        for delete in range(int(len(fittest) - (len(population)/2))):
            index = randint(0, len(fittest)-1) #remove a random ind from the fittest
            fittest.pop(0) #index 0 -> remove least fit from fittest
    max_pop_score.append(max(scores))
    mean_pop_score.append(av_score)
    return fittest


def get_os(parents):
    next_gen = []
    for par in range(os_amount-1):
        if par%2 == 0:
            p1 = parents[par]
            p2 = parents[par - 1]
            cross_point = randint(0, layers-1)
            os1 = p1[0: cross_point] + p2[cross_point: layers]
            os2 = p2[0: cross_point] + p1[cross_point: layers]
            next_gen.append(os1)
            next_gen.append(os2)
    return next_gen
                
def get_next(population):
    next_pop = []
    fittest = get_fit(population)
    for fit in fittest:
        next_pop.append(fit)
    offspring = get_os(fittest)
    for ofs in offspring:
        next_pop.append(ofs)
    next_pop_size = len(next_pop)
    extra = init_pop(population_size - next_pop_size)
    for ind in extra:
        next_pop.append(ind)
    return next_pop


def main():
    generation = 0
    pop = init_pop(population_size)
    while generation < max_gen:
        pop = get_next(pop)
        generation += 1
    print(best_track)
    plt.plot(max_pop_score)
    plt.plot(mean_pop_score)
    plt.show()


main()