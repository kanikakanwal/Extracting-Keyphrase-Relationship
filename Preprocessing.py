import sys 
import numpy 
reload(sys)  
sys.setdefaultencoding('utf8')
import os
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import nltk
import pickle
import gensim, logging
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#model = Sequential()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NoOfRows = 110

dictOfPosTags = {}
dictOfPosTags['CC'] = 0
dictOfPosTags['CD'] = 1
dictOfPosTags['DT'] = 2
dictOfPosTags['EX'] = 3
dictOfPosTags['FW'] = 4
dictOfPosTags['IN'] = 5
dictOfPosTags['JJ'] = 6
dictOfPosTags['JJR'] = 7
dictOfPosTags['JJS'] = 8
dictOfPosTags['LS'] = 9
dictOfPosTags['MD'] = 10
dictOfPosTags['NN'] = 11
dictOfPosTags['NNS'] = 12
dictOfPosTags['NNP'] = 13
dictOfPosTags['NNPS'] = 14
dictOfPosTags['PDT'] = 15
dictOfPosTags['POS'] = 16
dictOfPosTags['PRP'] = 17
dictOfPosTags['PRP$'] = 18
dictOfPosTags['RB'] = 19
dictOfPosTags['RBR'] = 20
dictOfPosTags['RBS'] = 21
dictOfPosTags['RP'] = 22
dictOfPosTags['TO'] = 23
dictOfPosTags['UH'] = 24
dictOfPosTags['VB'] = 25
dictOfPosTags['VBD'] = 26
dictOfPosTags['VBG'] = 27
dictOfPosTags['VBN'] = 28
dictOfPosTags['VBP'] = 29
dictOfPosTags['VBZ'] = 30
dictOfPosTags['WDT'] = 31
dictOfPosTags['WP'] = 32
dictOfPosTags['WP$'] = 33
dictOfPosTags['WRB'] = 34




# Used for taking each sentence as input and return the word2vec for each sentence
def word2vecMatrix(Sentence):
	global new_model
	sentenceList=Sentence.split()
	for i in range(len(sentenceList)):
		sentenceList[i] = sentenceList[i].strip('()-+=*&^%$#@!~`;:\'"?/.><,[]')
	sentenceMatrix=[]
	for s in sentenceList:
		if s  in new_model:
			wordMatrix=list(new_model[s])
			sentenceMatrix.append(wordMatrix)
		else:
			sentenceMatrix.append([0]*100)	

	length = len(sentenceMatrix)
	global NoOfRows
	for i in range(length,NoOfRows):
		sentenceMatrix.append([0]*100)	
	return sentenceMatrix


class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
 
	def __iter__(self):
		for fname in os.listdir(self.dirname):
			if fname.endswith(".txt"):
				for line in open(os.path.join(self.dirname, fname)):
					yield line.split()


def LoadModel():

	allSentences = MySentences('/home/kanika/Documents/Semester_3/DWDM/Major Project/scienceie2017_dev/dev')
	model = gensim.models.Word2Vec(allSentences)
	model.save('/tmp/mymodel')
	new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
	return new_model

def word2vec(sentence):
	
	return word2vecMatrix(sentence) 



def findListOfFiles(folderName):
	listOfANNFiles = []
	for file in os.listdir(folderName):
		if file.endswith(".ann"):
			listOfANNFiles.append(file)
	return listOfANNFiles


def sentencesOffset(filename,folderName):

	tokenizer = PunktSentenceTokenizer()
	filename = filename[:-3]+"txt"
	f = open(folderName+"/"+filename)
	dataOfFile = str(f.read())
	dataOfFile = dataOfFile.encode('ascii','ignore')
	sentences = tokenizer.tokenize(dataOfFile)
	offsetOfSentences = tokenizer.span_tokenize(dataOfFile)
	return sentences,offsetOfSentences

def findPositionVector(sentence,task,taskInformation):

	''' There were attempts to develop improved molecular 
	dynamics methods combining quantum features with a 
	semi classical treatment of dynamical correlations'''	
	word = taskInformation[task][2]	
	sentenceList = sentence.split()
	for i in range(len(sentenceList)):
		sentenceList[i] = sentenceList[i].strip('()-+=*&^%$#@!~`;:\'"?/.><,[]')
	lengthOfSentenceList = len(sentenceList)
	lengthOfWordList = len(word)
	global NoOfRows
	paddingLength = NoOfRows
	
	posVecWrtWord = [ [-50] for i in range(paddingLength)]
	#posVecWrtWord = list([list([-50])*paddingLength])
	j = 0
	p = 0
	i = 0
	start = 0
	end = lengthOfSentenceList
	while i < lengthOfSentenceList:
		j = 0
		if sentenceList[i] == word[j]:
			p=i
			j = 0
			while j < lengthOfWordList and sentenceList[i] == word[j]:
				j+=1
				i+=1
			if j == lengthOfWordList:
				start = p
				end = p+lengthOfWordList
				while p < end:
					posVecWrtWord[p][0]=0
					p+=1
				break
			else:
				i = p
		i+=1
	start-=1
	count = -1
	while start >= 0:
		posVecWrtWord[start][0]=count
		start-=1
		count-=1
	count=1
	while end < NoOfRows:
		posVecWrtWord[end][0]=count
		count+=1
		end+=1
	return posVecWrtWord


def findStatement(task1,task2,taskInformation,sentences,offsetOfSentences):

	offsetList1 = list(taskInformation[task1][1])
	offsetList2 = list(taskInformation[task2][1])
	offsetList1 = map(int,offsetList1)
	offsetList2 = map(int,offsetList2)
	offsets = offsetList1+offsetList2
	offsets.sort()
	offset=(offsets[0],offsets[3])
	
	for i in range(len(offsetOfSentences)):
		if offset[0] >= offsetOfSentences[i][0] and offset[1] <= offsetOfSentences[i][1]:
			return sentences[i]

def oneHotVector(sentence):

	global NoOfRows
	posTagMatrix = []
	sentenceList = sentence.split()
	for i in range(len(sentenceList)):
		sentenceList[i] = sentenceList[i].strip('()-+=*&^%$#@!~`;:\'"?/.><,[]')
	
	try:
		taggedWords = nltk.pos_tag(sentenceList)
	except:
		for i in range(0,NoOfRows):
			l = [0]*len(dictOfPosTags)
			posTagMatrix.append(l)	
		return posTagMatrix
	
	#code for taggedWords
	global dictOfPosTags
	for i in range(len(taggedWords)):
		l = [0]*len(dictOfPosTags)
		if len(sentenceList) > 0:
			if taggedWords[i][1] in dictOfPosTags:
				l[dictOfPosTags[taggedWords[i][1]]] = 1
				posTagMatrix.append(l)
			else:
				posTagMatrix.append(l)
		else:
			posTagMatrix.append(l)

	global NoOfRows
	length = len(posTagMatrix)
	for i in range(length,NoOfRows):
		l = [0]*len(dictOfPosTags)
		posTagMatrix.append(l)	
	return posTagMatrix



def readRelationDict(relationToTask,taskInformation,sentences,offsetOfSentences):

	
	global X_trainList,Y_trainList
	for eachRelation in relationToTask:
		listOfTasks = relationToTask[eachRelation]
		for eachPair in listOfTasks:
			task1 = eachPair[0]
			task2 = eachPair[1]
			sentence = findStatement(task1,task2,taskInformation,sentences,offsetOfSentences)
			if sentence != None:

				# call word to  vec for each sentence
				word2vecMatrix = word2vec(sentence)
				#print len(word2vecMatrix)
				# call feature vector(2-d matrix)
				posTagMatrix = oneHotVector(sentence)
				#print len(featureVector),
				# call position vector and feature vec for each sentence
				positionVector1 = findPositionVector(sentence,task1,taskInformation)
				positionVector2 = findPositionVector(sentence,task2,taskInformation)
				#print len(positionVector1),len(positionVector2)
				# send the 3 vectors to another function where we combine them.... this forms one element of X_train
				# each relation is one element of y_train				
				numpyWord2VecMatrixArray = numpy.array(word2vecMatrix)
				numpyPosTagMatrixArray = numpy.array(posTagMatrix)
				numpypositionVector1Array = numpy.array(positionVector1)
				numpypositionVector2Array = numpy.array(positionVector2)
				#print numpy.shape(numpyWord2VecMatrixArray)
				# print len(numpyPosTagMatrixArray)
				# print len(numpypositionVector1Array)
				# print len(numpypositionVector2Array)
				# print numpy.shape(numpyPosTagMatrixArray)
				# print numpy.shape(numpypositionVector1Array)
				# print numpy.shape(numpypositionVector2Array)
				
				X_trainElem = numpy.concatenate((numpyWord2VecMatrixArray,numpyPosTagMatrixArray,numpypositionVector1Array,numpypositionVector2Array),axis=1)				
				X_trainList.append(X_trainElem)
				Y_trainList.append(list([eachRelation]))

	

def readingEachANNFile( listOfANNFiles,folderName ):

	global X_trainList,Y_trainList
	for eachFIle in listOfANNFiles:
		taskToOffset = {}
		taskInformation = {}
		relationToTask = {}
		filename = folderName+"/"+eachFIle
		print filename
		f = open(filename,'r')
		sentences,offsetOfSentences = sentencesOffset(eachFIle,folderName)
		i = 1
		for eachLine in f:
			if "Hyponym-of" in eachLine:
				#lineSplit = eachLine.split() # R1	Hyponym-of Arg1:T20 Arg2:T7	
				lineSplit = eachLine.replace('\t',' ').split()
				task1 = lineSplit[2].split(':')[1]
				task2 = lineSplit[3].split(':')[1]
				if lineSplit[1] in relationToTask:	
					relationToTask[lineSplit[1]].append(tuple([task1,task2]))
				else:
					relationToTask[lineSplit[1]] = list()
					relationToTask[lineSplit[1]].append(tuple([task1,task2]))
				#findStatementWithTags(task1,task2,"Hyponym-of",taskToOffset,sentences,offsetOfSentences)

			elif "Synonym-of" in eachLine: # *	Synonym-of T7 T6
				lineSplit = eachLine.replace('\t',' ').split()
				task1 = lineSplit[2]
				task2 = lineSplit[3]
				if lineSplit[1] in relationToTask:	
					relationToTask[lineSplit[1]].append(tuple([task1,task2]))
				else:
					relationToTask[lineSplit[1]] = list()
					relationToTask[lineSplit[1]].append(tuple([task1,task2]))
				#findStatementWithTags(task1,task2,"Synonym-of",taskToOffset,sentences,offsetOfSentences)		
				
			else:	# T3	Process 65 79	thermalization
				lineSplit = eachLine.replace('\t',' ').split()
				taskInformation[lineSplit[0]] =  tuple([lineSplit[1],tuple([lineSplit[2],lineSplit[3]]),lineSplit[4:]])
				taskToOffset[lineSplit[0]] = (lineSplit[2],lineSplit[3])

		
		readRelationDict(relationToTask,taskInformation,sentences,offsetOfSentences)
		
		
i = 0
Y_trainList = []
X_trainList = []
folderName = "/home/kanika/Documents/Semester_3/DWDM/Major Project/scienceie2017_dev/dev"
new_model = LoadModel()
readingEachANNFile(findListOfFiles(folderName),folderName)	
dictOfTrain = {}
dictOfTrain["X_train"] = X_trainList
dictOfTrain["Y_train"] = Y_trainList
with open('test.pickle', 'wb') as handle:
  pickle.dump(dictOfTrain, handle)
