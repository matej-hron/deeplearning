import os

def mkdir(path):    
    if not os.path.isdir(path):
        os.mkdir(path)

def test(path):    
    print(path)

