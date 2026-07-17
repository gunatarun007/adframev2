import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("adframe.world_model.scene_database")

class SceneDatabase:
    """
    Lightweight local database manager that indexes and reads past execution
    runs and handles persistent histories.
    """
    def __init__(self, db_path: str = "./outputs/scene_database.json"):
        self.db_path = db_path
        self.records: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.records = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load scene database: {e}")
                self.records = []

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scene database: {e}")

    def insert_run(self, scene_id: str, frame_idx: int, metrics: Dict[str, Any]):
        self.records.append({
            "scene_id": scene_id,
            "frame_idx": frame_idx,
            "metrics": metrics
        })
        self.save()
