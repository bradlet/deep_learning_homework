import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np

BATCH_SIZE = 1
LEARNING_RATE = .003
EPOCHS = 6
WITH_POLLUTION = True

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

model_1 = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
)

model_2 = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
)

# For #3.c
model_3 = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.Sigmoid(),
    nn.Linear(1024, 1024),
    nn.Sigmoid(),
    nn.Linear(1024, 10),
)


# Model Training and Eval on test set at the end of each epoch
def run_model(n_net):
    optimizer = optim.SGD(n_net.parameters(), lr=LEARNING_RATE, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for predata, labels in trainloader:
            # Need to grab batch size from first dim, b/c the last batch in dataset might not be
            # The same size as BATCH_SIZE
            current_batch_size = predata.size()[0]
            # Alter the way we view our data so that matrix multiplication can work in spite of change
            # in batch size.
            data = predata.view_as(torch.zeros(current_batch_size, 784))

            optimizer.zero_grad()
            output = n_net(data)
            loss = loss_fn(output.view_as(torch.zeros(BATCH_SIZE, 10)), labels)
            loss.backward()
            optimizer.step()

        # Model evaluation at the end of each epoch
        correct = 0
        for predata, labels in testloader:
            current_batch_size = predata.size()[0]

            data = predata.view_as(torch.zeros(current_batch_size, 784))
            output = n_net(data)

            correct += sum(torch.where(torch.argmax(output, dim=1) == labels, 1, 0))

        print("CORRECT: ", correct)
        print("LEN TEST: ", len(testset))
        print("ACCURACY: ", correct.item() / len(testset))


if __name__ == '__main__':
    # transforms
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=transform)

    # Assuming random distribution of labels in data, should be able to just randomize 10% of data labels
    # we get as input, and that should adequately pollute our training data.
    if WITH_POLLUTION:
        trainset = list(trainset)

        for i in range(0, int(len(trainset)*.1)):
            trainset[i] = (trainset[i][0], np.random.randint(0, 9))

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    run_model(model_3)
