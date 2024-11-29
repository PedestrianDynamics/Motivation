[![motivation-app](https://github.com/PedestrianDynamics/Motivation/actions/workflows/streamlit-actions.yml/badge.svg)](https://github.com/PedestrianDynamics/Motivation/actions/workflows/streamlit-actions.yml)
  
# Motivation  

## Usage

### Setup virtual env

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Run simulation in a browser

```bash
streamlit run app.py
```

### Run simulation in a terminal:

1. Generate variations
```bash
python generage_variations.py --inifile files/inifile.json --param motivation_parameters/width --values 1.0,2.0
```

2. Run simulation with variations 

```bash
python simulation.py --variations-file variations.json --inifile inifile.json
```

with the json file being produced in step 1.



