from SceneRecognizer import *

my_svm = SceneRecognizer(None)
score = my_svm.cross_validate(['X2.txt','Y2.txt'])
score = my_svm.train(1000.0, ['X2.txt','Y2.txt'])
print(score)
