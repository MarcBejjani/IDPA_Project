from contentHelper import *
from FileNames import fileNames
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from math import log10


def getContentIndexingTable(tfidf):
    table = {}
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    for file in fileNames:
        doc = ET.parse(file)
        for el in doc.iter():
            stemmedTag = ps.stem(el.tag)
            if stemmedTag not in table:
                table[stemmedTag] = {}
                table[stemmedTag][file] = 0
            elif file not in table[stemmedTag]:
                table[stemmedTag][file] = 0
            table[stemmedTag][file] += 1
            if el.text is not None:
                string = el.text.split()
                for word in string:
                    if word not in stopWords:
                        stemmed = ps.stem(word)
                        if stemmed not in table:
                            table[stemmed] = {}
                            table[stemmed][file] = 0
                        elif file not in table[stemmed]:
                            table[stemmed][file] = 0
                        table[stemmed][file] += 1
    if tfidf == 2:
        for el in table:
            N = len(fileNames) + 1
            occurrences = len(table[el])
            for file in table[el]:
                table[el][file] *= log10(N / occurrences)
    return table


def filterDocs(query, indexingTable):
    queryVector = vectorParse(query)
    hits = []
    for el in queryVector:
        if el in indexingTable:
            for file in indexingTable[el]:
                if file not in hits:
                    hits.append(file)
    return hits


def filterFlatTextDocs(query, indexingTable):
    queryVector = parseTextQuery(query)
    hits = []
    for el in queryVector:
        if el in indexingTable:
            for file in indexingTable[el]:
                if file not in hits:
                    hits.append(file)
    return hits


def getCorpusTable(indexingTable, hits):
    table = {}
    for file in hits:
        table[file] = {}
        for element in indexingTable:
            if file in indexingTable[element]:
                table[file][element] = indexingTable[element][file]
            else:
                table[file][element] = 0
    return table


def parseQuery(query):
    vector = {}
    doc = ET.parse(query)
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    for el in doc.iter():
        stemmedTag = ps.stem(el.tag)
        if stemmedTag not in vector:
            vector[stemmedTag] = 0
        vector[stemmedTag] += 1
        if el.text is not None:
            string = el.text.split()
            for word in string:
                if word not in stopWords:
                    stemmed = ps.stem(word)
                    if stemmed not in vector:
                        vector[stemmed] = 0
                    vector[stemmed] += 1
    return vector



def parseTextQuery(query):
    string = query.split()
    vector = {}
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    for word in string:
        if word not in stopWords:
            stemmed = ps.stem(word)
            vector[stemmed] = string.count(word)
    return vector


def normalizeVectors(docA, docB):
    vectorA = docA
    vectorB = docB
    for el in vectorA:
        if el not in vectorB:
            vectorB[el] = 0
    for el in vectorB:
        if el not in vectorA:
            vectorA[el] = 0
    return vectorA, vectorB


def contentQueryDocument(query, corpusTable):
    similarities = {}
    for file in corpusTable:
        doc = parseQuery(query)
        normalizedQuery, normalizedDoc = normalizeVectors(doc, corpusTable[file])
        similarities[file] = cosineMeasure(normalizedQuery, normalizedDoc)
    return similarities


def queryFlatText(query, corpusTable):
    similarities = {}
    for file in corpusTable:
        normalizedQuery, normalizedDoc = normalizeVectors(query, corpusTable[file])
        similarities[file] = cosineMeasure(normalizedQuery, normalizedDoc)
    return similarities

