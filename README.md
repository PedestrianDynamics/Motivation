![build-badge](https://github.com/PedestrianDynamics/Motivation/actions/workflows/pylint.yml/badge.svg)
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

```bash
./run.sh simulation.py <input-file> <output-file.txt> <run-jpsvis>
```

- `python-script.py`: Model logic.
- `input-file.json`: inifile of the project with all necessary configs.
- `output_file.txt`: a successful simulation creates a trajectory file.
- `run-jpsvis (1|0)`: to visualize the trajectories with jpsvis



