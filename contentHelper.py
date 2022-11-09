import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math


def vectorParse(document):
    doc = ET.parse(document)
    vector = {}
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    for el in doc.iter():
        if el.tag in vector:
            vector[el.tag] += 1
        else:
            vector[el.tag] = 1

        if el.text is not None:
            string = el.text.split()
            for word in string:
                if word not in stopWords:
                    stemmed = ps.stem(word)
                    if stemmed in vector:
                        vector[stemmed] += 1
                    else:
                        vector[stemmed] = 1
    return vector


def normalizeVectors(vectorA, vectorB):
    elementsA = []; elementsB = []
    for x in vectorA:
        elementsA.append(x)
    for x in vectorB:
        elementsB.append(x)

    for el in elementsA:
        if el not in vectorB:
            vectorB[el] = 0
    for el in elementsB:
        if el not in vectorA:
            vectorA[el] = 0

    return vectorA, vectorB


def TF_IDF(vectorA, vectorB):
    newA = {}
    newB = {}

    for el in vectorA:
        if vectorA[el] != 0 and vectorB[el] != 0:
            newA[el] = vectorA[el] * math.log10(3/2)
            newB[el] = vectorB[el] * math.log10(3/2)
        elif vectorA[el] == 0 or vectorB[el] == 0:
            if vectorA[el] != 0:
                newA[el] = vectorA[el] * math.log10(3 / 1)
                newB[el] = 0
            elif vectorB[el] != 0:
                newB[el] = vectorB[el] * math.log10(3 / 1)
                newA[el] = 0
    return newA, newB


def cosineMeasure(vectorA, vectorB):
    num = 0
    denumA = 0
    denumB = 0
    for el in vectorA:
        num += (vectorA[el] * vectorB[el])
        denumA += vectorA[el] ** 2
        denumB += vectorB[el] ** 2
    denum = math.sqrt(denumA * denumB)
    return num/denum

