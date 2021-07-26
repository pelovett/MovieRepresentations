"""
A minimal class to unify APIs across different recommendation models. 
"""


class HostedModel:
    def __init__(self, version_string: str):
        self.version = version_string

    def predict(self, user_info: dict):
        raise NotImplementedError

    def load(self, model_file_path: str):
        raise NotImplementedError

    def __str__(self):
        return f"Hosted Model: {self.version}"
