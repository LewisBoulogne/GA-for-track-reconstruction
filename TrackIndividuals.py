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

# all variable parameters 
tracks = 2
layers = 5
population_size = 20
max_gen = 1000
os_amount = int(population_size/4)
fit_size = int(population_size/2)

# storing data to plot evolution
max_pop_score = [0]
mean_pop_score =[0]
best_track = [0]


# creating a randomly initialized population
def init_pop(size):
    pop = []
    for i in range(size):
        ind = []
        for layer in range(layers):
            ind.append((layer, randint(0, tracks - 1)))
        pop.append(ind)
    return pop


# find angle between two points !! 2 DIMENSIONAL !!
def get_angle(a, b, c):
    # using this equation to find the angle: AB.BC = len(AB).len(BC).cos(angle)
    AB_v = (a[0] - b[0], a[1] - b[1])
    BC_v = (b[0] - c[0], b[1] - c[1])
    AB_dot_BC = np.dot(AB_v, BC_v) # AB[0]*BC[0] + AB[1]*BC[1]
    len_AB = np.linalg.norm(AB_v) # np.sqrt((AB_v[0]**2) + (AB_v[1]**2))
    len_BC = np.linalg.norm(BC_v) # np.sqrt((BC_v[0]**2) + (BC_v[1]**2))
    if (len_AB * len_BC) == 0:
        return 0
    else:
        return AB_dot_BC / (len_AB * len_BC) # cosine of the angle


# get the angle score of an individual
def angle_score(track):
    a_score = 0
    # track's score will increase by 1 for each straight angle 
    # and decrease by 1 for each none-straight angle
    for hit in range(len(track)-1):
        if hit != 0: 
            # checking if the angle between three consecutive hits is 180 deg
            if get_angle(track[hit - 1], track[hit], track[hit + 1]) == 1: 
                a_score += 1
            else:
                a_score -= 1
    return a_score


# extracting the fittest indivduals of the population
def get_fit(population):
    global best_track
    scores = []
    # storing the scores of all tracks
    for ind1 in population:
        ind_score = angle_score(ind1)
        scores.append(ind_score)
        # finding the track with the best score to track evolution
        if ind_score > best_track[0]:
            best_track = [ind_score, ind1]
    av_score = np.mean(scores)
    fittest = []
    # fittest are all tracks that have a score greater the mean score
    for ind2 in population:
        if angle_score(ind2) >= av_score:
            fittest.append(ind2)
    fittest.sort()
    # controlling population size
    if len(fittest) > fit_size:
        for delete in range(int(len(fittest) - (len(population)/2))):
            index = randint(0, len(fittest)-1) #remove a random ind from the fittest
            fittest.pop(0) #index 0 -> remove least fit from fittest
    # keeping track of evolution
    max_pop_score.append(max(scores))
    mean_pop_score.append(av_score)
    return fittest


# creating the off-springs
def get_os(parents):
    next_gen = []
    # crossover reproduction
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


# creating the next generation          
def get_next(population):
    next_pop = []
    fittest = get_fit(population)
    for fit in fittest:
        next_pop.append(fit)
    offspring = get_os(fittest)
    for ofs in offspring:
        next_pop.append(ofs)
    next_pop_size = len(next_pop)
    # adding randomly initialized track with remaining room in population
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