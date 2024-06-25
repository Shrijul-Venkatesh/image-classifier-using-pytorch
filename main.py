import torch
import torch.optim as Optim
import torch.nn as NN
import matplotlib.pyplot as Plot

import dataset
import loader
import cnn_model

from utils.training_utils import train
from utils.testing_utils import test


def main():
    train_data = dataset.trainingDataset()
    test_data = dataset.testingDataset()

    # Uncomment and run the below script to verify if the dataset has been downloaded successfully

    # print(train_data.data.shape())
    # print(train_data.data.shape())

    loaders = loader.loader(train_data, test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn_model.CNN().to(device)

    optimizer = Optim.Adam(model.parameters(), lr=0.001)
    loss_function = NN.CrossEntropyLoss()

    for epoch in range(1, 11):
        train(model, device, loaders, optimizer, loss_function, epoch)
        test(model, device, loaders, loss_function)

    model.eval()
    data, target = test_data[0]
    data = data.unsqueeze(0).to(device)
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()
    print(f"Prediction : {prediction}")

    image = data.squeeze(0).squeeze(0).cpu().numpy()
    Plot.imshow(image, cmap="gray")
    Plot.show()


if __name__ == "__main__":
    main()
