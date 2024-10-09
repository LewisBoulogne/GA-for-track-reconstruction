import matplotlib.pyplot as plt
import numpy as np

from testing_toy_model import event as event

hits = event.hits
#print(event.modules[0].hits)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def segments():
    seg = []
    for i in range(len(event.modules) - 1):
        this_layer = []
        for hit1 in event.modules[i].hits:
            for hit2 in event.modules[i + 1].hits:
                this_layer.append(((hit1.x, hit1.y, hit1.z), (hit2.x, hit2.y, hit2.z)))
        seg.append(this_layer)
    return seg
all_seg = segments()


def plot_segments(array):
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

plot_segments(all_seg)

def plot_tracks(array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ((x1, y1, z1), (x2, y2, z2)) in array:
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'ro-')  

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Segments')
    plt.grid(True)
    plt.show()



print(all_seg)
print(all_seg[0])

def is_straight(seg1, seg2):
    isit = False
    #seg = ((x1, y1, z1), (x2, y2, z2))
    #vector_seg = (X, Y, Z)
    vector_seg1 = (seg1[0][0] - seg1[1][0], seg1[0][1] - seg1[1][1], seg1[0][2] - seg1[1][2])
    vector_seg2 = (seg2[0][0] - seg2[1][0], seg2[0][1] - seg2[1][1], seg2[0][2] - seg2[1][2])
    len_v1 = np.linalg.norm(vector_seg1)
    len_v2 = np.linalg.norm(vector_seg2)
    cos_a = np.dot(vector_seg1, vector_seg2)/(len_v1*len_v2)
    if cos_a == 1:
        isit = True
    return isit
        

def find_candidates():
    candidates = []
    #all_seg[0] = first layer segments
    for anchor in all_seg[0]:
        for segment in all_seg[1]:
            if is_straight(anchor, segment):
                candidates.append((anchor[0], segment[1]))
    return candidates
cand = find_candidates()


plot_tracks(cand)

def find_tracks():
    tracks = cand
    for candidate in tracks:
        for hit in event.hits:
            x = candidate[0][0]/hit.x
            y = candidate[0][1]/hit.y
            z = candidate[0][2]/hit.z
            t = candidate[0][0]/candidate[1][0]
            if x == t and y == x and z == x:
                candidate[1] = (hit.x, hit.y, hit.z)
    return tracks
tracks = find_tracks()

print(tracks)
plot_tracks(tracks)
