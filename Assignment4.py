import torch
import torchvision

# Constants
BATCH_SIZE = 30


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# Datasets
trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

# Data Loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if __name__ == "__main__":

    data, labels = next(iter(trainloader))
    data, labels = data.to(device), labels.to(device)

    print("Data Size: ", data.shape)
    print("Label Size: ", labels.shape)
