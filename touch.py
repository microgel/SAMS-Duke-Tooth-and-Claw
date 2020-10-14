import os
def touch(workingDir):
    if not os.path.exists(workingDir):
        os.mkdir(workingDir)