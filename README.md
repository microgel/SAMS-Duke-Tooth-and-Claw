# SAMS
www.github.com/RRavier/SAMS
www.github.com/ToothAndClaw/SAMS

SAMS: Surface Analysis, Mapping, and Statistics

This is a package containing software that allows for simultaneous matching of surfaces as well as a degree of statistical analysis. This package is suitable for rigid, homeomorphic, simply connected surfaces like anatomical surfaces and not for other datasets of interests such as SCAPE, FAUST, or non-homeomorphic or non-simply connected data.
## Installation and Setup
To install the necessary dependancies, run
```
pip install -r requirements.txt
```

To set variables and folder paths, go to `Setup/MappingSetup.py`. Upon initial installation, you must set the dataPath and projectDir paths.
## Runnning 
To run the script, simply run
```
python3 Execute01_RunPreparation.py
```
This will currently only run one mesh, if you wish to run more comment out the break in `Setup/PreparationScripts.py`. This will return the visualization each time, so be careful running this with multiple meshes or comment out the visualization script. 
