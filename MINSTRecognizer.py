import os

import model
import torch
from torchvision import transforms

class MINSTRecognizer:
    def __init__(self):
        model_file_path = 'model_params.pth'
        self.model = model.SimpleMLP()
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        if os.path.exists(model_file_path):
            self.model.load_state_dict(torch.load('model_params.pth'))
        self.model.eval()

    def recognize(self, image_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        image_data = self.transform(image_data)
        image_data = image_data.to(device)
        output = self.model(image_data)
        return output.argmax(dim=1).item()