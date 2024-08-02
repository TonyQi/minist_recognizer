import os

import torch
import MINSTDataSet
import model
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

model_file_path = 'model_params.pth'
model = model.SimpleMLP()

if os.path.exists(model_file_path):
    model.load_state_dict(torch.load('model_params.pth'))

model.eval()

x_train = MINSTDataSet.load_mnist_images('./train-images-idx3-ubyte')
y_train = MINSTDataSet.load_mnist_labels('./train-labels-idx1-ubyte')

index = random.randrange(10000)

test_image = x_train[index]
text_label = y_train[index]

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

test_image = transform(test_image)
plt.imshow(test_image[0], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_image = test_image.to(device)

output = model(test_image)



print(output.argmax().item())