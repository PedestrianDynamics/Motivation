![build-badge](https://github.com/PedestrianDynamics/Motivation/actions/workflows/pylint.yml/badge.svg)
# Motivation  

## Usage

```bash
./run.sh <python-script> <input-file> <output-file.txt> <run-jpsvis>
```

- `python-script.py`: Model logic
- `input-file.json`: inifile of the project with all necessary configs.
- `output_file.txt`: a successful simulation creates a trajectory file.
- `run-jpsvis (1|0)`: to visualize the trajectories with jpsvis

`run.sh` has three hard-coded values for other jpscore-related scripts.
