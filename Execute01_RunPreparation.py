import sys
import os
import json
import csv

import pandas as pd
import Setup.PreparationScripts as prep

from Setup.MappingSetup import ForceFeatureRecomputation, ForceDataRecomputation, dataPath
from Setup.initialize import workingPath
from datetime import datetime

def saveJson(path, data):
    with open(workingPath + path, 'w') as outfile:
        json.dump(data, outfile)
        
startTime = datetime.now()

print('Initializing Data Collection')
if not os.path.exists(workingPath):
    os.mkdir(workingPath)
    
Flags = {}
if os.path.exists(workingPath + 'Flags.json'):
    with open(workingPath + 'Flags.json') as json_file:
        Flags = json.load(json_file)
else:
    with open(workingPath + 'Flags.json', 'w+') as db_file:
            db_file.write(json.dumps({'Created' : True}))

Names = []
if os.path.exists(workingPath + 'Names.csv'):
    with open(workingPath + 'Names.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            Names = row

    if Names and not ForceDataRecomputation:
        print('Data already exists in working folder, will only reprocess if forced')
    else:
        prep.MoveDataToOutputFolder(dataPath, workingPath)
else:
    prep.MoveDataToOutputFolder(dataPath, workingPath)
    
print('Checking validity of Surfaces')
if 'AreHomeomorphic' in Flags:
    if Flags['AreHomeomorphic'] == 0 or ForceDataRecomputation:
        result = prep.HomeomorphismCheck(workingPath)
        if result['isDisc'] == -1:
            Flags['AreHomeomorphic'] = 0
            saveJson("Flags.json", Flags)
            raise Exception('Script stopping, please fix problems with meshes and then try again.')
        else:
            Flags['AreHomeomorphic'] = 1
            Flags['isDisc'] = result['isDisc']
            saveJson("Flags.json", Flags)
    else:
        print('Already verified homeomorphisms')
else:
    result = prep.HomeomorphismCheck(workingPath)
    if result['isDisc'] == -1:
        Flags['AreHomeomorphic'] = 0
        saveJson("Flags.json", Flags)
        raise Exception('Script stopping, please fix problems with meshes and then try again.')
    else:
        Flags['AreHomeomorphic'] = 1
        Flags['isDisc'] = result['isDisc']
        saveJson("Flags.json", Flags)

print('Computing Necessary Mesh Features')
if 'FeaturesComputed' in Flags and not ForceFeatureRecomputation:
    if Flags['FeaturesComputed'] == 1:
        print('Features are already computed, you may safely abort')
    else:
        prep.ComputeFeatures(workingPath)
else:
    prep.ComputeFeatures(workingPath)
Flags['FeaturesComputed'] = 1
saveJson("Flags.json", Flags)

print('Finished preparing meshes, you may begin mapping now')
print('Execute01 took:', datetime.now() - startTime)