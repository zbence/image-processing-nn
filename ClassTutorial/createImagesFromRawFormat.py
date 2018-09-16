import pandas as pd
import numpy as np
from PIL import Image

test_data = pd.read_csv(".\MNIST_fashion_raw\\fashion-mnist_test.csv")

print("Test data dimensions: ",  test_data.shape)

test_matrix = test_data.values
test_labels = test_matrix[:,0]
test_pictures = test_matrix[:,1:]


"""for i in range(test_pictures.shape[0]):
    raw_image = Image.fromarray(test_pictures[i].reshape(28,28).astype("uint8"))
    raw_image.save("fashion_mnist_train\\Picture" + str(i)+ ".jpeg")
"""
np.save("fashion_mnist_test\\test_labels.npy",test_labels)
