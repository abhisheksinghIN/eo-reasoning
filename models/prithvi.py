import torch
from transformers import AutoModel


MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"


class PrithviModel:

    def __init__(self, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

    @torch.inference_mode()
    def encode(self, x):

        x = x.to(self.device)

        output = self.model(
            pixel_values=x
        )

        return output.last_hidden_state
