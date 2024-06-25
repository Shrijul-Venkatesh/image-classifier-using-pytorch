import torch.nn as NN
import torch.nn.functional as Functional


class CNN(NN.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convLayer1 = NN.Conv2d(1, 10, kernel_size=5)
        self.convLayer2 = NN.Conv2d(10, 20, kernel_size=5)
        self.convLayer2_Dropout = NN.Dropout2d()

        self.fullyConnectedLayer1 = NN.Linear(320, 50)
        self.fullyConnectedLayer2 = NN.Linear(50, 10)

    def forward(self, data):
        data = Functional.relu(Functional.max_pool2d(self.convLayer1(data), 2))
        data = Functional.relu(
            Functional.max_pool2d(self.convLayer2_Dropout(self.convLayer2(data)), 2)
        )

        data = data.view(-1, 320)
        data = Functional.relu(self.fullyConnectedLayer1(data))

        data = Functional.dropout(data, training=self.training)

        data = self.fullyConnectedLayer2(data)

        return data
