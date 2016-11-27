from gist import *

directories = ['/home/servicerobot2/catkin_ws/src/scene_recognition/data2/hallways','/home/servicerobot2/catkin_ws/src/scene_recognition/data2/offices','/home/servicerobot2/catkin_ws/src/scene_recognition/data2/classrooms']
X, Y = create_matrices(directories)
np.savetxt('X2.txt', X)
np.savetxt('Y2.txt', Y)
