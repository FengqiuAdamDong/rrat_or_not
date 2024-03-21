import sys
import os

def count_files(path):
    files = os.listdir(path)
    return len(files)

files = []
for path in sys.argv[1:]:
    files.append(count_files(path))
print(sum(files))
