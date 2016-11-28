from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
#from sklearn.neural_network import MLPClassifier

import pickle
import numpy

def loadPickle(pickleName):
	
	with open(pickleName, 'rb') as handle:
  		b = pickle.load(handle)
  	X_train = numpy.array(b['X_train'])
  	Y_train = numpy.array(b['Y_train'])
  	return X_train,Y_train



X_train,Y_train = loadPickle("data.pickle")
X_test,Y_test = loadPickle("test.pickle")

oneDarray = list()
for each2dMatrix in X_train:
	oneDarray.append(each2dMatrix.ravel())


X_train = numpy.array(oneDarray)

oneDarray = list()
for each2dMatrix in X_test:
	oneDarray.append(each2dMatrix.ravel())

X_test = numpy.array(oneDarray)

oneDarray = list()
for each1dMatrix in Y_train:
	oneDarray.append(each1dMatrix[0])

Y_train = numpy.array(oneDarray)

oneDarray = list()
for each1dMatrix in Y_test:
	oneDarray.append(each1dMatrix[0])

Y_test = numpy.array(oneDarray)

# print numpy.shape(Y_test)
# print numpy.shape(Y_train)
clf = svm.SVC()

clf.fit(X_train,Y_train)

Y_out = clf.predict(X_test) 
#print "*******************************"
#print Y_out
#print "*******************************"
#print Y_test

correctlyPreditedCount = 0
for i in range(len(Y_out)):
	if Y_out[i] == Y_test[i]:
		correctlyPreditedCount += 1

# correctly Predicted Count
# totalPredictedSample
# totalGroundTruthSamples

totalPredictedSample = len(Y_test)

precision = correctlyPreditedCount/float(totalPredictedSample)

totalGroundTruthSamples = len(Y_test)

recall = correctlyPreditedCount/float(totalGroundTruthSamples)

fScore = 2 * precision * recall/(precision+recall)

print "Precision = ",precision
print "Recall = ",recall
print "F-score = ",fScore

# nnModel = MLPClassifier(solver='adam', alpha = 1e-5, random_state = 0)
# nnModel.fit(X_train, Y_train)
# Y_out = nnModel.predict(X_test)

# correctlyPreditedCount = 0
# for i in range(len(Y_out)):
# 	if Y_out[i] == Y_test[i]:
# 		correctlyPreditedCount += 1

# # correctly Predicted Count
# # totalPredictedSample
# # totalGroundTruthSamples

# totalPredictedSample = len(Y_out)

# precision = correctlyPreditedCount/float(totalPredictedSample)

# totalGroundTruthSamples = len(Y_train)

# recall = correctlyPreditedCount/float(totalGroundTruthSamples)

# fScore = 2 * precision * recall/(precision+recall)

# print precision,recall,fScore
