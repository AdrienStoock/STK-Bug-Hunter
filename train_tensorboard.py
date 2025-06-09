from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Répertoire où TensorBoard va chercher les fichiers
log_dir = "./logs/"
run_name = "ppo_test_2"  # Nom de l'expérience
model_save_path = os.path.join(log_dir, run_name, "trained_model")

# Créer l'environnement
env = make_vec_env("CartPole-v1", n_envs=1)

# Créer et entraîner le modèle
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Entraîner le modèle
model.learn(total_timesteps=30000, tb_log_name=run_name)

# Sauvegarder le modèle
print(f"\nSauvegarde du modèle dans {model_save_path}")
model.save(model_save_path)



