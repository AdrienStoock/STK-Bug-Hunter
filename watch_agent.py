import gymnasium as gym
from stable_baselines3 import PPO

def regarder_agent_entraine(chemin_modele, nb_episodes=5):
    """
    Charge un agent PPO entraîné et l'observe sur plusieurs épisodes.

    Args:
        chemin_modele (str): chemin vers le modèle entraîné (sans .zip)
        nb_episodes (int): nombre d'épisodes à exécuter
    """
    # Charger le modèle
    modele = PPO.load(chemin_modele)

    # Créer l'environnement avec rendu graphique
    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(nb_episodes):
        observation, info = env.reset()
        recompense_episode = 0
        termine = False
        tronque = False

        print(f"\n🎮 Épisode {episode + 1}")

        while not (termine or tronque):
            action, _ = modele.predict(observation)
            observation, recompense, termine, tronque, info = env.step(action)
            recompense_episode += recompense

        print(f"🏁 Récompense totale : {recompense_episode}")

    env.close()

if __name__ == "__main__":
    
    chemin_modele = "./logs/ppo_test_2/trained_model"
    regarder_agent_entraine(chemin_modele)
