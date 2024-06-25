# Digit Recognition and Image Classifier

A Python-based deep learning project for recognising digits in an image.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Features

- Utilizes Convolutional Neural Networks (CNNs) for image classification.
- Model training is performed on the MNIST datase of handwritten digits
- Uses matplotlib to represent the image during testing for easier user interaction and analysis.

## Installation

To install the necessary dependencies, you can use pip:

```bash
pip install -r requirements.txt

```

## Usage

Uncomment the

```python
print(train_data.data.shape())
print(train_data.data.shape())
```

code provided in the main.py file and execute it using the following command :

```bash
python main.py

```

This should first download the MNIST dataset into a folder called data. You should also get 60000 and 10000 as the shapes of the training and testing datasets respectively.

## Dataset

This project uses the TensorFlow MNIST dataset of handwritten images, each of size 28x28 pixels.
Link : https://www.tensorflow.org/datasets/catalog/mnist

## Contributing

Contributions are most welcome! You can contact me at [venkateshshrijul@gmail.com](mailto:venkateshshrijul@gmail.com) to discuss further

## License

This project is licensed under the MIT License. See the LICENSE file for more details
