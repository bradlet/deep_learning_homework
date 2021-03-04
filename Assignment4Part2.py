import torch
import torch.nn as nn
import numpy as np
from time import time
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Assignment4Part1 import row_accuracy

# Using code for auto encoder from https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html


mnist_data = datasets.KMNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:4096]


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs


if __name__ == "__main__":
    model = Autoencoder()
    max_epochs = 5
    outputs = train(model, num_epochs=max_epochs)

    embedding = model.encoder(outputs[max_epochs-1][1]).detach().numpy().squeeze()

    k_means = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)

    start_time = time()
    estimator = make_pipeline(StandardScaler(), k_means).fit(embedding)
    time_to_fit = time() - start_time
    print("Time to fit (ms): ", time_to_fit*1000)

    predictions = estimator[-1].labels_
    accuracy_table = row_accuracy(predictions[:100])
    for i in range(1, 10):
        accuracy_table = np.vstack((accuracy_table, row_accuracy(predictions[i*100:(i+1)*100])))

    print(accuracy_table)
