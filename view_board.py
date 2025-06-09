import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_scalars(log_dir, tags_to_plot=None):
    """
    Lit les fichiers TensorBoard dans log_dir, récupère les scalars
    spécifiés dans tags_to_plot et trace les courbes.
    
    :param log_dir: chemin vers le dossier contenant les fichiers TensorBoard
    :param tags_to_plot: liste des noms de scalars à tracer (ex : ['rollout/ep_rew_mean', 'train/loss'])
    """
    if not os.path.exists(log_dir):
        print(f"Le dossier {log_dir} n'existe pas.")
        return

    # Charge les événements TensorBoard
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Affiche tous les tags scalars disponibles
    available_tags = ea.Tags().get('scalars', [])
    print(f"Tags scalars disponibles : {available_tags}")

    # Si on ne précise pas, on trace tout ce qui est disponible
    if tags_to_plot is None:
        tags_to_plot = available_tags

    plt.figure(figsize=(10, 6))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, tag in enumerate(tags_to_plot):
        if tag not in available_tags:
            print(f"Tag '{tag}' non trouvé dans les logs, skip.")
            continue
        
        scalar_events = ea.Scalars(tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]

        plt.plot(steps, values, label=tag, color=colors[i % len(colors)])

    plt.xlabel("Steps")
    plt.ylabel("Valeurs")
    plt.title("Courbes TensorBoard extraites")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Remplace ce chemin par ton dossier de logs TensorBoard
    log_directory = "./logs/ppo_test_2"

    # Tags à tracer, ou None pour tout tracer
    tags = [
        # Récompense moyenne par épisode - indique la performance globale de l'agent
        # Plus cette valeur augmente, meilleur est l'agent
        "rollout/ep_rew_mean",

        # Fonction de perte totale de l'entraînement
        # Devrait diminuer progressivement pendant l'entraînement
        "train/loss",

        # Perte d'entropie - mesure la diversité/exploration des actions
        # Une valeur plus élevée indique plus d'exploration
        "train/entropy_loss",

        # Taux d'apprentissage - vitesse à laquelle le modèle apprend
        # Généralement diminue au fil du temps pour une convergence stable
        "train/learning_rate",
    ]

    plot_tensorboard_scalars(log_directory, tags)

