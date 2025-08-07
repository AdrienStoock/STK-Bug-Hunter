# STK-Bug-Hunter

This project uses Reinforcement Learning (PPO algorithm) to train an AI agent to play a racing video game. The long-term goal is to use this agent to detect potential bugs or anomalies in tracks through gameplay.

To make it easy for you to get started with the project, follow the next step 


## Prepare the environment 

- [ ] [Install python3](https://www.python.org/downloads/)
- [ ] [Clone https](https://github.com/AdrienStoock/STK-Bug-Hunter.git) or [Clone ssh](git@github.com:AdrienStoock/STK-Bug-Hunter.git) 
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
python3 drl_stk.py train --track=olivermath 
```

### Play using the best saved model:
```
python3 drl_stk.py play_best_model --track=olivermath 
```

### Command line arguments  

- mode (mandatory): 
    - `train`: Allow the agent to train 
    - `play_best_model`: Allow to see the best model 
- `--track`(mandatory): Specify the track 
- `--render_mode`(Optionnal): Choose between None (default) or human for visualization

*example*:
```
python3 drl_stk.py train --track=olivermath --render_mode=human
```

### Tracks list

-`abyss`
-`black_forest`
-`candela_city`
-`cocoa_temple`
-`cornfield_crossing`
-`fortmagma`
-`gran_paradiso_island`
-`hacienda`
-`lighthouse`
-`mines`
-`minigolf`
-`olivermath`
-`ravenbridge_mansion`
-`sandtrack`-
-`scotland`
-`snowmountain`
-`snowtuxpeak`
-`stk_enterprise`
-`volcano_island`
-`xr591`
-`zengarden`

## AI Description
This project uses the PPO algorithm from Stable-Baselines3 to train the agent.

-Step 1 : Train the agent to play (1st version)

The reward $R_t$ at each time step $t$ is computed as:

$$
R_t = \frac{\Delta d}{3} + 0.05 \cdot v_t - 0.5 \cdot b_t
$$

where:

- $\Delta d = d_t - d_{t-1}$ : distance progressed between two steps
- $v_t$ : scalar speed at time $t$ (norm of the velocity vector)
- $b_t \in \{0, 1\}$ : bug detection indicator  
  - $b_t = 1$ if a bug is detected (e.g., agent is stuck)  
  - $b_t = 0$ otherwise

---

**Explanation:**

- $\frac{\Delta d}{3}$ : reward for forward progress
- $0.05 \cdot v_t$ : small bonus for maintaining speed
- $-0.5 \cdot b_t$ : penalty if the agent is detected as stuck

*The agent trained on the *olivermath* track is able to almost complete a full lap after 600k to 800k steps.*



-Step 2 (Upcoming): Train agent to detect bugs in tracks 


## Authors 
Adrien Stoock 


## Project status
Still improving
