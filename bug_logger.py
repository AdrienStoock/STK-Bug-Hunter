import json
import datetime
from typing import Dict, Any

class BugLogger:
    def __init__(self, log_file: str = "bug_logs.json"):
        self.log_file = log_file
        self.bugs = []
        self._load_existing_logs()

    def _load_existing_logs(self):
        try:
            with open(self.log_file, 'r') as f:
                self.bugs = json.load(f)
        except FileNotFoundError:
            self.bugs = []

    def _save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.bugs, f, indent=4)

    def log_bug(self, bug_type: str, kart_position: Dict[str, float], additional_info: Dict[str, Any] = None):
        """
        Enregistre un bug dans le fichier JSON.
        
        Args:
            bug_type (str): Type de bug (ex: "collision", "out_of_bounds", etc.)
            kart_position (Dict[str, float]): Position du kart au moment du bug {"x": float, "y": float, "z": float}
            additional_info (Dict[str, Any], optional): Informations supplémentaires sur le bug
        """
        bug_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "bug_type": bug_type,
            "kart_position": kart_position,
        }
        
        if additional_info:
            bug_entry["additional_info"] = additional_info

        self.bugs.append(bug_entry)
        self._save_logs()

    def get_all_bugs(self) -> list:
        """Retourne tous les bugs enregistrés"""
        return self.bugs

    def get_bugs_by_type(self, bug_type: str) -> list:
        """Retourne tous les bugs d'un type spécifique"""
        return [bug for bug in self.bugs if bug["bug_type"] == bug_type]

    def clear_logs(self):
        """Efface tous les logs"""
        self.bugs = []
        self._save_logs() 