import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 300

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((.13,), (.32,))]
)

train_set = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

training_error_history = []
test_error_history = []


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Dropout()
        )
        self.fullyConnected = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        # print(x.shape)
        for layer in self.convolutions:
            x = layer(x)
            plt.imshow(x[0][0].cpu().detach().numpy())
            plt.show()
            # print(layer._get_name(), ": ", x.shape)
        x = torch.flatten(x, 1)  # Keep
        for layer in self.fullyConnected:
            x = layer(x)
            # print(layer._get_name(), ": ", x.shape)
        # plt.imshow(x.cpu().detach().numpy())
        # plt.show()
        return x


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Dropout()
        )
        self.fullyConnected = nn.Sequential(
            nn.Linear(in_features=3072, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        x = torch.flatten(x, 1)  # Keep
        for layer in self.fullyConnected:
            x = layer(x)
        return x


def run_model(n_net, loss_fn):
    optimizer = optim.SGD(n_net.parameters(), lr=LEARNING_RATE, momentum=0.01)

    for epoch in range(EPOCHS+1):
        # TRAIN
        n_net.train()
        train_correct = 0
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            output = n_net(data)
            # Record train accuracy data
            output_classifications = torch.argmax(output, dim=1)
            train_correct += (output_classifications == labels).sum().item()
            # How output is used differs between CrossEntropyLoss and MSELoss
            if isinstance(loss_fn, nn.MSELoss):
                output_max = torch.max(output, dim=1)[0]
                loss = loss_fn(output_max, labels.float())
            else:  # defaulting to CrossEntropyLoss
                loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TEST
        n_net.eval()
        test_correct = 0
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            output = n_net(data)
            # Record test accuracy data
            output_classifications = torch.argmax(output, dim=1)  # dim=0 was the only time I saw change in test acc.
            test_correct += (output_classifications == labels).sum().item()

        # At end of epoch, record train and test error to plot.
        print("Epoch: ", epoch)
        print("train correct: ", train_correct, " | size trainset: ", len(train_set))
        print("test correct: ", test_correct, " | size testset: ", len(test_set))
        training_error_history.append(1-(train_correct/len(train_set)))
        test_error_history.append((1-(test_correct/len(test_set))))


if __name__ == "__main__":
    # model = NeuralNet()
    model = CustomNet()
    model.to(DEVICE)

    run_model(model, nn.CrossEntropyLoss())

    plt.plot(training_error_history, label='Training Error')
    plt.plot(test_error_history, label='Test Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
