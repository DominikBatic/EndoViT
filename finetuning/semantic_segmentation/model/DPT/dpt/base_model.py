import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        if (path):
            parameters = torch.load(path, map_location=torch.device("cpu"))

            if "model" in parameters:
                parameters = parameters["model"]
                print(f"Loaded model checkpoint from ... \"{path}\"", end="\n\n")

            msg = self.load_state_dict(parameters, strict=False)
            print(msg)
