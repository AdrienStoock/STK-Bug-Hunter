# ML_VideoGame

This project uses Reinforcement Learning (PPO algorithm) to train an AI agent to play a racing video game. The long-term goal is to use this agent to detect potential bugs or anomalies in tracks through gameplay.

To make it easy for you to get started with the project, follow the next step 


## Prepare the environment 

- [ ] [Install python3](https://www.python.org/downloads/)
- [ ] [Clone https](https://gitlab.com/Adrien-Stoock/ml_videogame.git) or [Clone ssh](git@gitlab.com:Adrien-Stoock/ml_videogame.git) 
- [ ] Create the venv:

```
python -m venv my_project

#for windows 
my_project_env\Scripts\activate

#for linux 
source my_project_env/bin/activate

#installation of prerequisites
pip install -r requirements.txt
```

## Running the code 

### Train an agent on a specific track (e.g., olivermath):
``` 
python3 drl_stk.py train --track olivermath 
``` 

### Play using the best saved model:
``` 
python3 drl_stk.py play_best_model --track olivermath 
``` 

### Command line arguments  

- mode (mandatory): 
    - train : Allow the agent to train 
    - play_best_model : Allow to see the best model 
- --track (mandatory): Specify the track 
- --render_mode (Optionnal): Choose between None (default) or human for visualization

## AI Description
This project uses the PPO algorithm from Stable-Baselines3 to train the agent.

-Step 1 (Upcoming): Train the agent to play  
-Step 2 (Upcoming): Train agent to detect bugs in tracks 


## Authors 
Adrien Stoock 


## Project status
Still improving
