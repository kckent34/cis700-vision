from PIL import Image
import leargist
import glob
import numpy as np

def gist_descriptors(im):
    return leargist.color_gist(im)


def resize_image(im):
    return im.resize((64,48), Image.ANTIALIAS)


def create_matrices(directories):
    category = 0
    #X = np.zeros((200,960))
    # Y = np.zeros(200)
    Y = []
    X = []
    for dir in directories:
        dir_jpg = dir + '/*.jpg'
        for filename in glob.glob(dir_jpg):
            im = Image.open(filename)
            resized_image = resize_image(im)
            desc = gist_descriptors(resized_image)
            X.append(desc)
            Y.append(category)
        category += 1

    return np.array(X), np.array(Y)

def append_matrices(directories, filepaths):
    X = np.loadtxt(filepaths[0])
    Y = np.loadtxt(filepaths[1])

#directories = ['/home/servicerobot2/catkin_ws/src/scene_recognition/data2/hallways','/home/servicerobot2/catkin_ws/src/scene_recognition/data2/offices','/home/servicerobot2/catkin_ws/src/scene_recognition/data2/classrooms']
#X, Y = create_matrices(directories)

#np.savetxt('X2.txt', X)
#np.savetxt('Y2.txt', Y)
