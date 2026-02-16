import matplotlib.pyplot as plt
import torch

def show_map(tensor, title=""):
    img = tensor.detach().cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.colorbar()
    plt.show()

