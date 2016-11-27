
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score

from gist import *


class SceneRecognizer:
    def __init__(self, classes):
        self.clf = None
        self.dir = classes
        self.class_maps = {0:'hallway',1:'office',2:'classroom'}
        self.current_descriptor = None

    def train(self, c_val, filename=None):
        if filename is None:
            #get matrices
            X,Y = create_matrices(self.dir)
            #create svm
        else:
            X_file = filename[0]
            Y_file = filename[1]
            X = np.loadtxt(X_file)
            Y = np.loadtxt(Y_file)

        self.clf = svm.SVC(decision_function_shape='ovo',probability=True, C=c_val)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        self.clf.fit(X_train, Y_train)
        joblib.dump(self.clf, 'SVM_model.pkl')
        score = self.clf.score(X_test, Y_test)
        # print('Accuracy was {:.4f}'.format(self.clf.score(X_test, Y_test)))
        return score

    def cross_validate(self, filename = None):
        if filename is None:
            #get matrices
            X,Y = create_matrices(self.dir)
            #create svm
        else:
            X_file = filename[0]
            Y_file = filename[1]
            X = np.loadtxt(X_file)
            Y = np.loadtxt(Y_file)

        self.clf = svm.SVC(decision_function_shape='ovo', probability=True)
        C_s = np.logspace(-10, 10, 10)

        scores = list()
        scores_std = list()
        for C in C_s:
            self.clf.C = C
            this_scores = cross_val_score(self.clf, X, Y, n_jobs=1)
            scores.append(np.mean(this_scores))
            scores_std.append(np.std(this_scores))

        # Do the plotting
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.semilogx(C_s, scores)
        plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
        plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
        locs, labels = plt.yticks()
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
        plt.ylabel('CV score')
        plt.xlabel('Parameter C')
        plt.ylim(0, 1.1)
        plt.show()

    def load_model(self, filename):
        self.clf = joblib.load(filename)

    def predict(self,im):
        resize_image(im)
        desc = gist_descriptors(im)
        desc = np.array(desc)
        self.current_descriptor = desc.reshape(1, -1)
        return self.class_maps[int(self.clf.predict(desc.reshape(1, -1)))]

    def decision(self):
        dec = self.clf.decision_function(self.current_descriptor)
        return dec

    def probs(self):
        probs = self.clf.predict_proba(self.current_descriptor)
        return probs


# def main():
#     my_svm = SceneRecognizer(None)
#     #score = my_svm.cross_validate(['X2.txt','Y2.txt'])
#     score = my_svm.train(['X2.txt','Y2.txt'])
#     print(score)
#
#
# main()





