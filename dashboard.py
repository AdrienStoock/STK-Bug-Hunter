# dashboard.py

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

class DashboardLogger:
    @staticmethod
    def find_latest_tensorboard_run(log_root: str) -> str:
        if not os.path.exists(log_root): return None
        subdirs = [os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
        if not subdirs: return None
        return max(subdirs, key=os.path.getmtime)

    @staticmethod
    def plot_tensorboard_scalars_single_plot(log_dir, output_dir, tags_to_plot=None):
        if not os.path.exists(log_dir): return
        os.makedirs(output_dir, exist_ok=True)

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        available_tags = ea.Tags().get('scalars', [])

        metric_groups = {
            "rollout": [
                "rollout/ep_len_mean",
                "rollout/ep_rew_mean",
                "rollout/mean_reward",
                "rollout/collisions",
                "rollout/off_tracks",
                "rollout/crashes",
                "rollout/blocked"
            ],
            "time": [
                "time/fps",
                "time/iterations",
                "time/time_elapsed",
                "time/total_timesteps"
            ],
            "train": [
                "train/approx_kl",
                "train/clip_fraction",
                "train/clip_range",
                "train/entropy_loss",
                "train/explained_variance",
                "train/learning_rate",
                "train/loss",
                "train/n_updates",
                "train/policy_gradient_loss",
                "train/value_loss"
            ]
        }

        for group_name, group_metrics in metric_groups.items():
            plt.figure(figsize=(15, 8))
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

            for i, tag in enumerate(group_metrics):
                if tag not in available_tags:
                    print(f"Tag '{tag}' non trouvé.")
                    continue

                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                plt.plot(steps, values, label=tag, color=colors[i % len(colors)])

            plt.xlabel("Steps")
            plt.ylabel("Valeurs")
            plt.title(f"{group_name} - Métriques")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            output_path = os.path.join(output_dir, f"{group_name}_metrics.png")
            plt.savefig(output_path)
            plt.close()
            print(f"✅ Graphique {group_name} sauvegardé dans : {output_path}")

        if "train/learning_rate" in available_tags:
            events = ea.Scalars("train/learning_rate")
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label="Learning Rate", color='b')
            plt.xlabel("Steps")
            plt.ylabel("Learning Rate")
            plt.title("Évolution du Learning Rate")
            plt.grid(True)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "learning_rate.png")
            plt.savefig(output_path)
            plt.close()
            print(f"✅ Graphique Learning Rate sauvegardé dans : {output_path}")

        if "train/n_updates" in available_tags:
            events = ea.Scalars("train/n_updates")
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label="Nombre de mises à jour", color='g')
            plt.xlabel("Steps")
            plt.ylabel("Nombre de mises à jour")
            plt.title("Évolution du nombre de mises à jour")
            plt.grid(True)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "n_updates.png")
            plt.savefig(output_path)
            plt.close()
            print(f"✅ Graphique Nombre de mises à jour sauvegardé dans : {output_path}")
