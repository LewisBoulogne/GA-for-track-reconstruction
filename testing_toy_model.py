import trackhhl.toy.simple_generator as toy
import matplotlib.pyplot as plt
import numpy as np

N_MODULES = 4
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0

detector = toy.SimpleDetectorGeometry(
    module_id=list(range(N_MODULES)),
    lx=[LX]*N_MODULES,
    ly=[LY]*N_MODULES,
    z=[i+Z_SPACING for i in range(N_MODULES)]
)

generator = toy.SimpleGenerator(
    detector_geometry=detector,
    theta_max=np.pi/6
)

N_PARTICLES = 4
event = generator.generate_event(N_PARTICLES)



#fig = plt.figure()
#x = fig.add_subplot(111, projection='3d')
#event.display(ax)
#plt.show()