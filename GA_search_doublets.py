import matplotlib.pyplot as plt
from random import randint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import trackhhl.toy.simple_generator as toy




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
                    this_layer.append(([hit1.x, hit1.y, hit1.z], [hit2.x, hit2.y, hit2.z]))
            seg.append(this_layer)
        return seg

    def plot_segments(self, array):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for layer in array:
            for ((x1, y1, z1), (x2, y2, z2)) in layer:
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'ro-')  
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Segments')
        plt.grid(True)
        plt.show()

    def plot_tracks(self, array, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ((x1, y1, z1), (x2, y2, z2)) in array:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'ro-')  
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
                    candidates.append([anchor[0], segment[1]])
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
                    t_x = (candidate[0][0]-hit.x)/(candidate[0][0]-candidate[1][0])
                    t_y = (candidate[0][1]-hit.y)/(candidate[0][1]-candidate[1][1])
                    t_z = (candidate[0][2]-hit.z)/(candidate[0][2]-candidate[1][2])
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
GENTEIC ALGORITHM
"""
#GA parameters:
population_size = 10
fit_size = population_size/2

max_pop_score = [0]
mean_pop_score =[0]
best_ind = [0, [0, 0, 0]]

def init(size):
    population = []
    for i in range(size):
        exp = randint(0, 10)
        rounded = randint(1, 10)
        individual = [i, 10**(-exp), rounded]
        population.append(individual)
    return population

def evaluate(ind):
    score = 0
    name = ind[0]
    angle_window = ind[1]
    dec_places = ind[2]
    this_agent = agent(angle_window, dec_places)
    agent_tracks = this_agent.find_tracks(event)
    if len(agent_tracks) == Particles:
        score += 1
    for track1 in agent_tracks:
        if len(track1) == layers - 1:
            score += 1
    for t1 in agent_tracks:
        for t2 in agent_tracks:
            for hit1 in t1:
                for hit2 in t2:
                    if hit1 == hit2:
                        score -= 1
    return score
"""
    for track in agent_tracks:
        for sol in tracks_1:
            if track == sol:
                score += 1
"""
    
def get_fit(population):
    global best_ind
    scores = []
    # storing the scores of all tracks
    for ind1 in population:
        ind_score = evaluate(ind1)
        scores.append(ind_score)
    av_score = np.mean(scores)
    fittest = []
    for i in population:
        this_score = evaluate(i)
        if this_score >= av_score:
            fittest.append([this_score, i])
    fittest.sort(key=lambda tup:tup[0], reverse=False)
    while len(fittest) > fit_size:
        fittest.pop(0)
    # uploading best track
    for fit in fittest:
            if fit[1] != best_ind[1] and fit[0] >= best_ind[0]:
                best_ind = fit
    # keeping track of evolution
    max_pop_score.append(max(scores))
    mean_pop_score.append(av_score)
    final_fittest = []
    for r in fittest:
        final_fittest.append(r[1])
    return final_fittest

Particles = 5
layers = 5
event = make_event(layers, Particles)
agent_1 = agent(1e-4, 3)
tracks_1 = agent_1.find_tracks(event)
#agent_1.plot_tracks(tracks_1, '3D tracks')

def main():
    POPULATION = init(population_size)
    GENERATION = 0
    while GENERATION < 100:
        fit = get_fit(POPULATION)
        other = init(population_size - len(fit))
        POPULATION = fit + other
        GENERATION += 1
    score = best_ind[0]
    ind = best_ind[1]
    best_agent = agent(ind[1], ind[2])
    best_agent.plot_tracks(best_agent.find_tracks(event), '3D tracks from best agent')

main()