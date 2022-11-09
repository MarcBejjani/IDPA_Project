from tkinter import filedialog
from tkinter import *
from StructureQueryHelper import *
from contentQueryHelper import *
from StructureHelper import *


root = Tk()
root.title('Query')
root.geometry("700x500")
root.configure(bg='black')
docA = ''
docB = ''
filename = ''
similarities = {}


def uploadFiles():
    global filename
    filename = filedialog.askopenfilename(title='Select file', filetypes=[('Xml Files', '*xml')])
    print(filename)


def uploadDocA():
    global docA
    docA = filedialog.askopenfilename(title='Select file', filetypes=[('Xml Files', '*xml')])


def uploadDocB():
    global docB
    docB = filedialog.askopenfilename(title='Select file', filetypes=[('Xml Files', '*xml')])


def takeInput():
    INPUT = my_entry.get('1.0', 'end-1c')


def queryFile():
    global similarities
    if var1.get() == 2:
        similarities = {}
        table = getIndexingTable(var3.get())
        hits = filterDocuments(filename, table)
        matrixList = getCorpusMatrix(filename, table, hits)
        similarities = queryDocument(filename, matrixList)
    if var1.get() == 1:
        similarities = {}
        table = getContentIndexingTable(var3.get())
        hits = filterDocs(filename, table)
        vectorTable = getCorpusTable(table, hits)
        similarities = contentQueryDocument(filename, vectorTable)
    Output.delete('1.0', END)
    if not my_entry.get():
        k = len(fileNames)
    else:
        k = int(my_entry.get())
    docs = sorted(similarities, key=similarities.get, reverse=True)[:k]
    for document in docs:
        file = document[document.index('\\') + 1:]
        Output.insert(END, f'{file}: {similarities[document]}\n')


def queryText():
    global similarities
    similarities = {}
    table = getContentIndexingTable(var3.get())
    hits = filterFlatTextDocs(textQuery.get(), table)
    query = parseTextQuery(textQuery.get())
    vectorTable = getCorpusTable(table, hits)
    similarities = queryFlatText(query, vectorTable)
    Output.delete('1.0', END)
    if not my_entry.get():
        k = len(fileNames)
    else:
        k = int(my_entry.get())
    docs = sorted(similarities, key=similarities.get, reverse=True)[:k]
    for document in docs:
        file = document[document.index('\\') + 1:]
        Output.insert(END, f'{file}: {similarities[document]}\n')


def compareDocs():
    if var2.get() == 1:
        vectorA = vectorParse(docA)
        vectorB = vectorParse(docB)
        vectorA, vectorB = normalizeVectors(vectorA, vectorB)
        if var3.get() == 2:
            vectorA, vectorB = TF_IDF(vectorA, vectorB)
        similarity = cosineMeasure(vectorA, vectorB)
        Output.delete('1.0', END)
        Output.insert(END, similarity)
    if var2.get() == 2:
        documentA = getMatrixModel(docA)
        documentB = getMatrixModel(docB)
        vectorA, vectorB = normalizeStructure(documentA, documentB)
        if var3.get() == 2:
            vectorA, vectorB = IDF(vectorA, vectorB)
        similarity = matrixCosineSimilarity(vectorA, vectorB)
        Output.delete('1.0', END)
        Output.insert(END, similarity)


XMLButton = Button(root, text='Choose XML File', font=("Arial", 15),  fg="black", bg="white", command=uploadFiles)
XMLButton.place(x=50, y=55)

docAButton = Button(root, text='Choose first file', font=("Arial", 15),  fg="black", bg="white", command=uploadDocA)
docAButton.place(x=850, y=55)
docBButton = Button(root, text='Choose second File', font=("Arial", 15),  fg="black", bg="white", command=uploadDocB)
docBButton.place(x=1010, y=55)
queryButton = Button(root, text='Get similarity', font=('Arial', 15), fg='black', bg='white', command=compareDocs)
queryButton.place(x=950, y=150)
Label(root, text="Compare documents",  font=("Arial", 20), bg="blue", fg="white").place(x=900, y=10)
var2 = IntVar()
Radiobutton(root, font=20, text="Content", variable=var2, value=1).place(x=900, y=100)
Radiobutton(root, font=20, text="Structure", variable=var2, value=2).place(x=1010, y=100)


docQueryButton = Button(root, text='Query Document', font=('Arial', 15), fg='black', bg='white', command=queryFile)
docQueryButton.place(x=50, y=150)


label = Label(root, text="Query XML file",  font=("Arial", 20), bg="blue",
              fg="white")
label.place(x=50, y=10)

var1 = IntVar()
Radiobutton(root, font=20, text="Content", variable=var1, value=1).place(x=50, y=100)
Radiobutton(root, font=20, text="Structure", variable=var1, value=2).place(x=160, y=100)



text = Label(root, text="Query flat text",  font=("Arial", 20), bg="blue", fg="white")
text.place(x=500, y=10)
textQuery = Entry(root, font=("Arial", 20), bg="blue", width=15)
textQuery.place(x=475, y=100)
textQueryButton = Button(root, text='Query Text', font=('Arial', 15), fg='black', bg='white', command=queryText)
textQueryButton.place(x=540, y=150)

my_entry = Entry(root, font=("Arial", 20), bg="blue", width=15)
my_entry.place(x=270, y=250)


var3 = IntVar()
Radiobutton(root, font=20, text="TF", variable=var3, value=1).place(x=600, y=250)
Radiobutton(root, font=20, text="TF_IDF", variable=var3, value=2).place(x=700, y=250)

Output = Text(root, width=100, height=50, bg='silver', fg='white', font= ('Arial', 20))
Output.place(x=0, y=300)


root.mainloop()
