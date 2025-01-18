import os

class Tools():
    def __init__(self):
        pass
    
    def getAbsPath(filePath):
        abs_path = os.path.abspath(filePath)
        print(abs_path)
        return abs_path

    def convert_slashes(directory_path):
        return directory_path.replace("\\", "/")


