# MovieRepresentations
The development of a recommendation engine for movies using the Netflix movie recommendation dataset.


## Installation
Install necessary packages (preferrably with a virtual environment) using the following bash command: 
```bash
pip3 install -r requirements.txt
```

## Serving A Model
To serve a model set the environment variable `MODEL_CHECKPOINT` before running the following command in the `movierep/` directory:
```bash
uvicorn RecModelServer:app
```
