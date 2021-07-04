# DATYS: Disambiguating API method mentions via Type  Scoping
## 1. Install Docker enviroment
```
$ cd datys
$ docker compose up -d
$ docker exec -it datys bash
```
## 2. Activate anaconda env inside docker container
```
$ conda activate env
```
## 3. Run experiment code
```
$ python src/run_experiment.py
```