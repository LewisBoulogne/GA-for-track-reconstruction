import matplotlib.pyplot as plt
from random import randint
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import trackhhl.toy.simple_generator as toy
import time
import tracemalloc




"""
AGENT CLASS

agent pararmeters:
angle window
extrapolate to 1 or more layers
confidence score for each track (create best fit line and assess average distance from the hits and line)
number of decimal points for rounding t in extrapolation
"""


class agent:
    def __init__(self, angle_window, extrapolation_uncertainty, extrapolation_range = 0):
        self.angle_window = angle_window
        self.extrapolation_uncertainty = extrapolation_uncertainty # rounding -> number of decimal points
        self.extrapolation_range = extrapolation_range # number of layers range
    
    def get_segments(self, event):
        seg = []
        for i in range(len(event.modules) - 1):
            this_layer = []
            for hit1 in event.modules[i].hits:
                for hit2 in event.modules[i + 1].hits:
                    this_layer.append([[hit1.x, hit1.y, hit1.z], [hit2.x, hit2.y, hit2.z]])
            seg.append(this_layer)
        return seg

    def plot_tracks(self, array, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if len(array) > 1:
            for track in array:
                for i in range(len(track) - 1):
                    ax.plot([track[i][0], track[i + 1][0]], [track[i][1], track[i + 1][1]], [track[i][2], track[i + 1][2]], 'ro-')
        if len(array) == 1:
            for i in range(len(array) - 1):
                    ax.plot([array[i][0], array[i + 1][0]], [array[i][1], array[i + 1][1]], [array[i][2], array[i + 1][2]], 'ro-')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.grid(True)
        plt.show()

    def is_straight(self, seg1, seg2):
        isit = False
        #seg = ((x1, y1, z1), (x2, y2, z2))
        #vector_seg = (X, Y, Z)
        vector_seg1 = [seg1[0][0] - seg1[1][0], seg1[0][1] - seg1[1][1], seg1[0][2] - seg1[1][2]]
        vector_seg2 = [seg2[0][0] - seg2[1][0], seg2[0][1] - seg2[1][1], seg2[0][2] - seg2[1][2]]
        len_v1 = np.linalg.norm(vector_seg1)
        len_v2 = np.linalg.norm(vector_seg2)
        cos_a = np.dot(vector_seg1, vector_seg2)/(len_v1*len_v2)
        if 1 - cos_a < self.angle_window :
            isit = True
        return isit
            
    def find_candidates(self, event):
        array = self.get_segments(event)
        candidates = []
        #all_seg[0] = first layer segments
        for anchor in array[0]:
            for segment in array[1]:
                if self.is_straight(anchor, segment):
                    candidates.append(anchor + segment)
        return candidates

    def find_tracks(self, event):
        array = self.find_candidates(event)
        for candidate in array:
            if self.extrapolation_range == 0:
                end = len(event.modules)
            else:
                end = 3 + self.extrapolation_range
            for layer in range(3, end):
                for hit in event.modules[layer].hits:
                    t_x = (candidate[0][0]-hit.x)/(candidate[0][0]-candidate[3][0])
                    t_y = (candidate[0][1]-hit.y)/(candidate[0][1]-candidate[3][1])
                    t_z = (candidate[0][2]-hit.z)/(candidate[0][2]-candidate[3][2])
                    if round(t_x, self.extrapolation_uncertainty) == round(t_y, self.extrapolation_uncertainty) == round(t_z, self.extrapolation_uncertainty):
                        candidate.append([hit.x, hit.y, hit.z])
        return array


def make_event(N_MODULES, N_PARTICLES):
    LX = float("+inf")
    LY = float("+inf")
    Z_SPACING = 1.0
    detector = toy.SimpleDetectorGeometry(
        module_id=list(range(N_MODULES)),
        lx=[LX]*N_MODULES,
        ly=[LY]*N_MODULES,
        z=[i+Z_SPACING for i in range(N_MODULES)])
    generator = toy.SimpleGenerator(
        detector_geometry=detector,
        theta_max=np.pi/6)
    ev = generator.generate_event(N_PARTICLES)
    return ev


"""
GENETIC ALGORITHM
"""
#GA parameters:
population_size = 10
fit_size = population_size/2

max_pop_score = [0]
mean_pop_score =[0]
best_ind = [0, [0, 0]]
runtimes = [0]

def init(size):
    population = []
    for i in range(size):
        exp = randint(0, 10)
        rounded = randint(1, 10)
        individual = [10**(-exp), rounded]
        population.append(individual)
    return population

def evaluate(ind):
    print("evaluating: " + str(ind))
    angle_window, dec_places = ind
    start_time = time.time()
    this_agent = agent(angle_window, dec_places)
    agent_tracks = this_agent.find_tracks(event)
    end_time = time.time()

    # found as many tracks as there are particles? 
    tracks_found_score = len(agent_tracks)/PARTICLES
    if tracks_found_score > 1:
        tracks_found_score = 0

    # how fast is the agent?
    agent_runtime = end_time - start_time
    runtimes.append(agent_runtime)
    runtime_score = max(1 - agent_runtime/np.mean(runtimes), 0) 
    # if runtime is bigger than average ratio will be negative therefore the score will be 0

    # are the tracks as long as the amount of layers?
    crosspoints = 0
    tracks_length = []
    for track1 in agent_tracks: 
        tracks_length.append(len(track1))
        # are there duplicate tracks?
        for track2 in agent_tracks: 
            if track1 == track2:
                crosspoints += 1 
    crosshits_score = max(1 - crosspoints/len(agent_tracks), 0)
    length_score = np.mean(tracks_length)/(LAYERS-1)
    if length_score > 1:
        length_score = 0
    """
    print("n tracks found: " + str(len(agent_tracks)))
    print("n tracks found score: " + str(tracks_found_score))
    print("runtime score: " + str(runtime_score))
    print("length_score: " + str(length_score))
    print("unique tracks score: " + str(crosshits_score))
    """
    score = (tracks_found_score + 
             runtime_score + 
             length_score + 
             crosshits_score)/4
    
    print("score: " + str(score))
    return score

    
def get_fit(population):
    global best_ind
    scores = []
    # storing the scores of all tracks
    for ind1 in population:
        ind_score = evaluate(ind1)
        scores.append(ind_score)
    av_score = np.mean(scores)
    print("av_score: " + str(av_score))
    fittest = []
    for i in population:
        this_score = evaluate(i)
        if this_score > av_score:
            fittest.append([this_score, i])
            print("added to fittest: " + str(i))
    print("n fit before: " + str(len(fittest)))
    fittest.sort(key=lambda tup:tup[0], reverse=False)
    while len(fittest) > fit_size:
        print("pop: " + str(fittest[0]))
        fittest.pop(0)
    # uploading best track
    for fit in fittest:
            if fit[1] != best_ind[1] and fit[0] >= best_ind[0]:
                print("NEW best individual: " + str(best_ind))
                best_ind = fit
    # keeping track of evolution
    final_fittest = []
    for r in fittest:
        final_fittest.append(r[1])
    return final_fittest

PARTICLES = 5
LAYERS = 10
event = make_event(LAYERS, PARTICLES)
agent_1 = agent(1e-4, 3)
tracks_1 = agent_1.find_tracks(event)
#print(tracks_1)
#agent_1.plot_tracks(tracks_1, '3D tracks')

def main(max_gen, population_size):
    POPULATION = init(population_size)
    GENERATION = 0
    while GENERATION < max_gen:
        print("gen: " + str(GENERATION))
        fit = get_fit(POPULATION)
        other = init(population_size - len(fit))
        print("n fit after: " + str(len(fit)))
        print()
        POPULATION = fit + other
        GENERATION += 1
    best_agent = agent(best_ind[1][0], best_ind[1][1])
    print()
    print("final best: " + str(best_ind))
    best_tracks = best_agent.find_tracks(event)
    print(str(len(best_tracks)) + " tracks found")
    print()
    print(best_tracks)
    #best_agent.plot_tracks(best_tracks, '3D tracks from best agent')

main(10, 6)

