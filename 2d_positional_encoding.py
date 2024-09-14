import imageio.v2 as imageio
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils


class MLP(nn.Module):
    def __init__(self, frequencies: torch.Tensor):
        super(MLP, self).__init__()
        self.frequencies = frequencies # (2, L)
        
        self.fc1 = nn.Linear(2*L, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 2)
        x_input = (x @ self.frequencies) * 2 * torch.pi                         # (batch_size, L)
        encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)  # (batch_size, 2*L)
        return encoding # (batch_size, 2*L)
    
    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


def predict_image(model, all_coords: torch.Tensor, image_shape: tuple, batch_size: int) -> torch.Tensor:
    # all_coords: (H*W, 2)
    predicted_pixels = torch.zeros((all_coords.shape[0], 3), device=all_coords.device)
    with torch.no_grad():
        for i in range(0, all_coords.shape[0], batch_size):
            predicted_pixels[i:i+batch_size] = model(all_coords[i:i+batch_size])
    predicted_image = predicted_pixels.reshape(image_shape)
    return predicted_image


def sample_batch(all_coords: torch.Tensor, all_pixels: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # all_coords: (H*W, 2)
    # all_pixels: (H*W, 3)
    indices = torch.randint(0, all_coords.shape[0], (batch_size,), device=image.device)     # (batch_size, )
    return all_coords[indices], all_pixels[indices] # (batch_size, 2), (batch_size, 3)


def train_model(model, image, optimizer, criterion, epochs, batch_size, device):
    scores = []

    x = torch.linspace(-1, 1, image.shape[0])
    y = torch.linspace(-1, 1, image.shape[1])
    all_coords = torch.stack(torch.meshgrid(x,y), dim=-1).reshape(-1, 2).to(device)
    all_pixels = image.reshape(-1, 3)

    model.train()
    pbar = tqdm(total=epochs)
    val_epoch = set(np.geomspace(1, epochs, num=50, endpoint=True).astype(int))
    for epoch in range(epochs):
        
        optimizer.zero_grad()
        coords_batch, pixel_batch = sample_batch(all_coords, all_pixels, batch_size)
        loss = criterion(model(coords_batch), pixel_batch)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        pbar.update(1)
        if (epoch+1) in val_epoch:
            model.eval()
            with torch.no_grad():
                predicted_image = predict_image(model, all_coords, image.shape, batch_size)
            
            loss = criterion(predicted_image, image)
            psnr = utils.psnr(image, predicted_image)
            print(f"PSNR: {psnr:.2f} dB")
            scores.append(psnr)
            plt.imsave(f"val_output/{(epoch+1):04}.jpg", predicted_image.cpu().numpy())
            model.train()

    pbar.close()
    return scores


if __name__ == '__main__':
    utils.set_seed(60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = "data/dryfruit.jpg"
    image = imageio.imread(image_path).astype(np.float32) / 255.0
    
    # Make square shape image of size 512 x 512 (only for visualization purposes)
    H, W, _ = image.shape
    cy, cx, res = H//2, W//2, min(H, W)//2
    image = image[cy-res:cy+res, cx-res:cx+res]
    image = cv2.resize(image, (512, 512))

    # Try experimenting with different values of L (number of frequencies) and m (freq_multiplier)
    L, m = 256, 10
    frequencies = m * torch.randn(2,L).to(device)       # (2, L)
    
    image = torch.from_numpy(image).to(device)
    model = MLP(frequencies).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, image, optimizer, criterion, epochs=4000, batch_size=3000, device=device)
    utils.create_gif('val_output/', 'output.gif', duration=10)

