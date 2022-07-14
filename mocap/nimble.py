import nimblephysics as nimble
import pprint
import ipdb
import time

path_osim = '/home/takaraet/gait_modeling/Rajagopal_Scaled_v2.osim'
file: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(path_osim)

path_mot = '/home/takaraet/gait_modeling/results_ik.mot'
mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(file.skeleton, path_mot)


pprint.pprint(mot.poses[:, 100])

world = nimble.simulation.World()
world.addSkeleton(file.skeleton)
gui = nimble.NimbleGUI(world)

gui.serve(8080)

for i in range(1000):
    file.skeleton.setPositions(mot.poses[:, i])
    gui.nativeAPI().renderSkeleton(file.skeleton)
    time.sleep(.5)

gui.blockWhileServing()




#ipdb.set_trace()
print(file.skeleton.getDofs()[0].getName())
print("\n\n")
file.skeleton.setPositions(mot.poses[:, 0])

dict = file.skeleton.getJointWorldPositionsMap()

pprint.pprint(dict)


