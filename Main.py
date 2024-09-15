import matplotlib.pyplot as plt

from random import randint

"""
LEVEL 1: generate tracks randomly until they correspond to the solution
"""

generation = 0
tracks = []
solution = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)]
]


def init():
    global tracks
    for r in range(2):
        tracks.append([(randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2))])


def generate():
    removed = 0
    for c in solution:
        for r in tracks:
            if r != c:
                tracks.remove(r)
                removed += 1
    for i in range(removed):
        tracks.append([(randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2))])


def found():
    # check whether the tracks correspond to solution
    global tracks, solution
    if tracks == solution:
        return True
    else:
        return False


def main_lvl1():
    global generation
    generation = 1
    solve = False
    init()
    while not solve:
        if found():
            solve = True
            print("!!!SOLVED!!!")
            print("After " + str(generation) + " generations")
            print("solution")
            for c in solution:
                print(c)
            print("tracks")
            for i in tracks:
                print(i)

        else:
            print("gen " + str(generation))
            generate()
            generation += 1
            print("solution")
            for c in solution:
                print(c)
            print("tracks")
            for i in tracks:
                print(i)


"""
LEVEL 2: generate 10 tracks and keep the tracks that correspond to a track from the solution
"""

def initialize():
    global tracks
    for r in range(10):
        tracks.append([(randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2)),
                       (randint(0, 2), randint(0, 2))])

def evaluate(num):
    global tracks, solution
    new_tracks = []
    removed = 0
    if num == 0:
        for i in solution:
            for j in tracks:
                if j == i:
                    new_tracks.append(j)
                else:
                    tracks.remove(j)
                    removed += 1



def main_lvl2():
    global generation
    generation = 1
    solved = 0
    init()
    while solved != 2:
        evaluate(solved)
