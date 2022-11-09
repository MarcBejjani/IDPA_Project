import xml.etree.ElementTree as ET
from math import log10, sqrt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def getMatrixModel(document):
    doc = ET.parse(document)
    root = doc.getroot()
    matrix = {}
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    parentMap = {c: p for p in doc.iter() for c in p}

    for el in doc.iter():
        if el.text is not None:
            string = el.text.split()
            if string:
                currentNode = el
                currentPath = el.tag
                while currentNode != root:
                    currentNode = parentMap[currentNode]
                    currentPath = currentNode.tag + "/" + currentPath

            for word in string:
                if word not in stopWords:
                    stemmed = ps.stem(word)
                    if currentPath not in matrix:
                        matrix[currentPath] = {}
                    if word not in matrix[currentPath]:
                        matrix[currentPath][stemmed] = 0
                    matrix[currentPath][stemmed] = string.count(word)

    return matrix


def normalizeStructure(documentA, documentB):
    for path in documentA:
        if path not in documentB:
            documentB[path] = {}
            for element in documentA[path]:
                documentB[path][element] = 0
    for path in documentB:
        if path not in documentA:
            documentA[path] = {}
            for element in documentB[path]:
                documentA[path][element] = 0
    return documentA, documentB


def IDF(vectorA, vectorB):
    for path in vectorA:
        for word in vectorA[path]:
            if vectorA[path][word] > 0 and vectorB[path][word] > 0:
                vectorA[path][word] *= log10(3/2)
                vectorB[path][word] *= log10(3/2)
            elif vectorA[path][word] == 0 or vectorB[path][word] == 0:
                if vectorA[path][word] != 0:
                    vectorA[path][word] *= log10(3/1)
                    vectorB[path][word] = 0
                elif vectorB[path][word] != 0:
                    vectorB[path][word] *= log10(3/1)
                    vectorA[path][word] = 0
    return vectorA, vectorB


def matrixCosineSimilarity(matrixA, matrixB):
    num = 0
    denum1 = 0
    denum2 = 0
    for path in matrixA:
        for node in matrixA[path]:
            num += matrixA[path][node] * matrixB[path][node]
            denum1 += matrixA[path][node]**2
            denum2 += matrixB[path][node]**2

    denum = sqrt(denum1*denum2)

    return num/denum
