import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from FileNames import fileNames
from math import log10, sqrt


def getIndexingTable(tfidf):
    table = {}
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))

    for file in fileNames:
        doc = ET.parse(file)
        root = doc.getroot()
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
                            path = currentPath + f'/{stemmed}'
                            if path not in table:
                                table[path] = {}
                                table[path][file] = 0
                            elif file not in table[path]:
                                table[path][file] = 0
                            table[path][file] += 1
    if tfidf == 2:
        for el in table:
            N = len(fileNames) + 1
            occurrences = len(table[el])
            for file in table[el]:
                table[el][file] *= log10(N / occurrences)

    return table


def getCorpusMatrix(query, indexingTable, hits):
    matrixList = {}
    for file in hits:
        matrixList[file] = {}
        for element in indexingTable:
            if file in indexingTable[element]:
                matrixList[file][element] = indexingTable[element][file]
            else:
                matrixList[file][element] = 0
    return matrixList


def getMatrixModelQuery(document):
    doc = ET.parse(document)
    root = doc.getroot()
    terms = []
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))

    for el in doc.iter():
        if el.text is not None:
            string = el.text.split()
            if string:
                currentNode = el
                currentPath = el.tag
                while currentNode != root:
                    currentNode = doc.find(f'.//{currentNode.tag}/..')
                    currentPath = currentNode.tag + "/" + currentPath
                for word in string:
                    if word not in stopWords:
                        stemmed = ps.stem(word)
                        path = currentPath + f'/{stemmed}'
                        terms.append(path)
    matrix = {}
    for term in terms:
        if term not in matrix:
            matrix[term] = 0
        matrix[term] += 1

    return matrix


def filterDocuments(query, indexingTable):
    vectorQuery = getMatrixModelQuery(query)
    hits = []
    for element in vectorQuery:
        if element in indexingTable:
            for file in indexingTable[element]:
                if file not in hits:
                    hits.append(file)
    return hits


def normalizeDoc(matrixA, matrixB):
    newA = matrixA
    newB = matrixB

    for nodeB in newB:
        if nodeB not in newA:
            newA[nodeB] = 0
    for nodeA in newA:
        if nodeA not in newB:
            newB[nodeA] = 0
    return newA, newB


def queryCosineSimilarity(matrixA, matrixB):
    num = 0
    denumA = 0
    denumB = 0
    for el in matrixA:
        num += (matrixA[el] * matrixB[el])
        denumA += matrixA[el] ** 2
        denumB += matrixB[el] ** 2
    denum = sqrt(denumA * denumB)
    return num / denum


def queryDocument(document, corpusMatrix):
    similarities = {}
    for matrix in corpusMatrix:
        docMatrix = getMatrixModelQuery(document)
        docMatrixNormalized, corpusMatrixNormalized = normalizeDoc(docMatrix, corpusMatrix[matrix])
        similarities[matrix] = queryCosineSimilarity(docMatrixNormalized, corpusMatrixNormalized)
    return similarities

