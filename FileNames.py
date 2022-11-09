import os

directory = 'samples'
fileNames = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        fileNames.append(f)
