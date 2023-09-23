#necessary liblaries to import in part 1
import os
import glob
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
#ignoring some numpy errors
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


#paths of image directories
trainpath = "Vegetable Images/train"
validationpath = "Vegetable Images/validation"
testpath = "Vegetable Images/test"

#All Layers in data
class_dict={}
for i , folder in enumerate(os.listdir(trainpath)):
    class_dict[folder] = i

class_dict


# a function to obtain images from directory
# returns preprocessed data and label matrixes to use in neural network progress
def read_images(path, class_dict, size=(60, 60), convert_gray=True):
    x = []
    y = []
    # for each file directory
    for folder in os.listdir(path):
        # for each file
        for file_path in glob.glob(str(path + '/' + folder + '/*.jpg')):
            img = Image.open(file_path)  # openning image
            if convert_gray:
                img = img.convert('L')  # grayscale image
            img = img.resize(size)  # resizing image
            img = np.array(img)  # converting image to numpy array
            x.append(img)
            y.append(class_dict[folder])

    x = np.array(x)
    y = np.array(y)

    # normalization
    x = x.astype('float32') / 255.0

    # one-hot-encoding of label values
    y_processed = np.zeros((15, x.shape[0]))
    for col in range(x.shape[0]):
        val = y[col]
        for i in range(15):
            if (val == i):
                y_processed[val, col] = 1

    x = x.reshape(x.shape[0], -1).T  # flatten process
    return x, y_processed

#reading our train , validation and test data
x_train, y_train = read_images(trainpath, class_dict, size=(60,60), convert_gray=True)
x_validation, y_validation = read_images(validationpath, class_dict, size=(60,60), convert_gray=True)
x_test, y_test = read_images(testpath, class_dict, size=(60,60), convert_gray=True)

#train , validation and test matrixes shapes
print("Train Data Shape : ",x_train.shape)
print("Train Label Shape : ", y_train.shape)
print("Validation Data Shape : ", x_validation.shape)
print("Validation Label Shape : ", y_validation.shape)
print("Test Data Shape : ", x_test.shape)
print("Test Label Shape : ", y_test.shape)

#shuffling train data
keys = np.array(range(x_train.shape[1]))
np.random.shuffle(keys)
x_train = x_train[:, keys]
y_train = y_train[:, keys]

# reverse layer list
reverse_class_list = {}
for key in class_dict:
    reverse_class_list[class_dict[key]] = key

# visualizing some random images from our data
fig = plt.figure(figsize=(16, 16))
for i in range(1, 37):
    img_index = np.random.randint(0, x_train.shape[1])
    fig.add_subplot(6, 6, i)
    plt.imshow(x_train[:, img_index].reshape((60, 60)), cmap='gray')
    plt.axis('off')
    plt.title(reverse_class_list[np.argmax(y_train[:, img_index], axis=0)])


#activation functions
#for input and hidden layers
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.maximum(0 , x)

#for output layer
def softmax(x):
    eX = np.exp(x- np.max(x))
    return eX/eX.sum(axis =0 ,keepdims=True)

#derivate of activation functions
def der_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def der_tanh(z):
    return 1-tanh(z)*tanh(z)

def der_relu(z):
    return np.greater(z, 0)


# functions for saving and loading our trained models later use in test data
def save_model(weights, biases, path):
    with open("models/" + path + ".npy", 'wb') as f:
        np.save(f, weights)
        np.save(f, biases)


def load_model(path):
    with open("models/" + path + ".npy", 'rb') as f:
        weights = np.load(f, allow_pickle=True)
        biases = np.load(f, allow_pickle=True)
        return weights, biases


# function to initialize starter parameters
def initilaize_biases_and_weights(layers):
    # The list 'layers' contains the number of neurons in the respective layers.
    # For example, if 'layers' is [10,5,1] ,then the input layer has 10 neurons ,
    # the output layer has 1 neuron and network contains 1 hidden layer which has 5 neurons.

    biases = []
    weights = []

    for i in range(1, len(layers)):
        # initilaize biases to 0
        biases.append(np.zeros((layers[i], 1)))
        # initilaize weights to random numbers using normal distribution
        weights.append(np.random.randn(layers[i], layers[i - 1]) * 0.01)
    return weights, biases


def forward_prop(X_train, weights, biases, activation_function):
    activation_func_list = {'sigmoid': sigmoid, 'ReLU': ReLU, 'relu': ReLU, 'tanh': tanh}  # activation functions
    forward_activation = activation_func_list[activation_function]  # used activation function

    forward_results = []  # results array
    A = X_train
    k = len(weights) - 1  # number of layers without output layer
    for i in range(k):
        Z = np.dot(weights[i], A) + biases[i]  # Z = XW+B
        # user choice activation function used in input layer and hidden layers
        A = forward_activation(Z)  # A = activation_function(Z)
        forward_results.append({'z': Z, 'a': A})  # append to the result array

    # softmax function used in output layer
    Z = np.dot(weights[k], A) + biases[k]
    A = softmax(Z)
    forward_results.append({'z': Z, 'a': A})
    return forward_results


def cost_function(A, y_train):
    # length of y_train
    m = y_train.shape[1] if y_train.shape[1] > 0 else 1
    # cost = - np.sum( np.multiply(np.log(A),y_train)) / m
    # negative log-likelihood function
    cost = -np.sum(np.multiply(np.log(A), y_train) + np.multiply((1 - y_train), np.log(1 - A))) / m
    return cost


def backward_prop(X_train, y_train, weights, biases, forward_results, batch_size, activation_function):
    activation_func_list = {'sigmoid': der_sigmoid, 'ReLU': der_relu, 'relu': der_relu,
                            'tanh': der_tanh}  # derivates of activation functions
    backward_activation = activation_func_list[activation_function]  # user choice derivative function

    L = len(forward_results) - 1  # number of layers except output layer
    backward_results = [0] * (L + 1)  # resulting array
    dZ = forward_results[L]['a'] - y_train  # output layer derivative calculated
    # derivatie of weights
    if (L == 0):
        dW = (1.0 / batch_size) * np.dot(dZ, X_train.T)
    else:
        dW = (1.0 / batch_size) * np.dot(dZ, forward_results[L - 1]['a'].T)
    # derivative of biases
    dB = (1.0 / batch_size) * np.sum(dZ, axis=1, keepdims=True)

    backward_results[L] = {'dW': dW, 'dB': dB}

    for i in reversed(range(L)):  # start from last hidden layer to input layer
        dA = np.dot(weights[i + 1].T, dZ)

        dZ = dA * backward_activation(forward_results[i]['a'])

        # derivatie of weights
        if (i == 0):
            dW = (1.0 / batch_size) * np.dot(dZ, X_train.T)
        else:
            dW = (1.0 / batch_size) * np.dot(dZ, forward_results[i - 1]['a'].T)

        dB = (1.0 / batch_size) * np.sum(dZ, axis=1, keepdims=True)  # derivative of biases

        backward_results[i] = {'dW': dW, 'dB': dB}

    return backward_results


def update_weights_and_biases(weights, biases, backward_results, learning_rate):
    for i in range(len(weights)):
        # update weights and biases with backward results and learning rate
        biases[i] = biases[i] - learning_rate * backward_results[i]['dB']
        weights[i] = weights[i] - learning_rate * backward_results[i]['dW']
    return weights, biases


def predict(X_test, y_test, weights, biases, activation_function='ReLU'):
    forward_results = forward_prop(X_test, weights, biases, activation_function)
    predictions = np.argmax(forward_results[-1]['a'], axis=0)
    labels = np.argmax(y_test, axis=0)
    return forward_results, predictions, labels


def train(X_train, Y_train, X_valid, Y_valid, layer_dimns, activation_function='ReLU', epochs=5, batch_size=128,
          learning_rate=0.02):
    print('Training begins...')
    st = time.time()

    # arrays for plotting total training loss, validation loss and validation accuracy
    loss_arr = []
    valid_loss_arr = []
    valid_acc_arr = []
    last_Accuracy = 0  # last accuracy value for early stopping

    weights, biases = initilaize_biases_and_weights(layer_dimns)  # initialize weights and biases
    batches = -(-X_train.shape[1] // batch_size)  # number of batches

    for i in range(epochs + 1):
        # shuffling train data
        keys = np.array(range(X_train.shape[1]))
        np.random.shuffle(keys)
        X_train = X_train[:, keys]
        Y_train = Y_train[:, keys]
        # for early stopping
        weightsPrev, biasesPrev = weights.copy(), biases.copy()

        total_loss = 0
        for j in range(batches):
            # data sample for training
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            x = X_train[:, begin:end]
            y = Y_train[:, begin:end]
            batch_len = end - begin
            # forward propagation
            forward_results = forward_prop(x, weights, biases, activation_function)
            # backward propagation
            backward_results = backward_prop(x, y, weights, biases, forward_results, batch_len, activation_function)
            # updating weights and biases
            weights, biases = update_weights_and_biases(weights, biases, backward_results, learning_rate)
            # add to the total cost
            total_loss += cost_function(forward_results[-1]['a'], y)
        loss_arr.append(total_loss)

        # calculate validation accuracy and loss
        forward_results_valid, predictions, labels = predict(X_valid, Y_valid, weights, biases, activation_function)
        new_acc = accuracy_score(predictions, labels) * 100
        loss_valid = cost_function(forward_results_valid[-1]['a'], Y_valid)

        valid_acc_arr.append(new_acc)
        valid_loss_arr.append(loss_valid)
        # every 10 epoch, display the total training loss, validation loss and validation accuracy
        if i % 10 == 0:
            print("""Epoch {}: Total Training Loss = {:.3f} , Validation Loss = {:.3f} , Validation Accuracy = {:.3f}
            """.format(i, total_loss, loss_valid, new_acc))

            # Early Stopping Applied
            if new_acc < last_Accuracy:
                print('Early Stopping applied.')
                print_plots(loss_arr, valid_loss_arr, valid_acc_arr)
                return weightsPrev, biasesPrev

            last_Accuracy = new_acc

    print('Training done in {:.3f} seconds \n'.format(time.time() - st))

    print_plots(loss_arr, valid_loss_arr, valid_acc_arr)
    return weights, biases


# fuction to plot total training loss, validation loss and validation accuracy
def print_plots(loss_arr, valid_loss_arr, valid_acc_arr):
    fig = plt.figure(figsize=(14, 14))
    fig.add_subplot(4, 3, 1)
    plt.plot(loss_arr)
    plt.title('Total Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    fig.add_subplot(4, 3, 2)
    plt.plot(valid_loss_arr)
    plt.title('Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    fig.add_subplot(4, 3, 3)
    plt.plot(valid_acc_arr)
    plt.title('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

#input and output layer shapes
input_layer = x_train.shape[0] #input layer
output_layer = y_train.shape[0] #output layer

layer_dims_0 = [input_layer , output_layer ]  # No hidden layers
layer_dims_1 = [input_layer, 144 , output_layer ]  # 1 hidden layer
layer_dims_2 = [input_layer, 144 , 64 , output_layer ]  # 2 hidden layers

allModelNames_0hl = []
print('Model with No Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_sigmoid_16bs_005lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005 ',
     'model_0hl_sigmoid_16bs_005lr'])

print('Model with No Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_tanh_16bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005 ',
                          'model_0hl_tanh_16bs_005lr'])

print('Model with No Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_relu_16bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ',
                          'model_0hl_relu_16bs_005lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_sigmoid_64bs_005lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ',
     'model_0hl_sigmoid_64bs_005lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_tanh_64bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005 ',
                          'model_0hl_tanh_64bs_005lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_relu_64bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ',
                          'model_0hl_relu_64bs_005lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_sigmoid_128bs_005lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ',
     'model_0hl_sigmoid_128bs_005lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_tanh_128bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ',
                          'model_0hl_tanh_128bs_005lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_0hl_relu_128bs_005lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ',
                          'model_0hl_relu_128bs_005lr'])

print('Model with No Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_sigmoid_16bs_02lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ',
     'model_0hl_sigmoid_16bs_02lr'])

print('Model with No Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_tanh_16bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ',
                          'model_0hl_tanh_16bs_02lr'])

print('Model with No Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_relu_16bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ',
                          'model_0hl_relu_16bs_02lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_sigmoid_64bs_02lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ',
     'model_0hl_sigmoid_64bs_02lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_tanh_64bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ',
                          'model_0hl_tanh_64bs_02lr'])

print('Model with No Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_relu_64bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ',
                          'model_0hl_relu_64bs_02lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_sigmoid_128bs_02lr')
allModelNames_0hl.append(
    ['Model with No Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ',
     'model_0hl_sigmoid_128bs_02lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_tanh_128bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02',
                          'model_0hl_tanh_128bs_02lr'])

print('Model with No Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_0, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_0hl_relu_128bs_02lr')
allModelNames_0hl.append(['Model with No Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ',
                          'model_0hl_relu_128bs_02lr'])

print('Each models Total Training Loss, Validation Loss and Validation Accuracy plots respectively \n')

allModelNames_1hl = []
print('Model with one Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_sigmoid_16bs_005lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005 ',
     'model_1hl_sigmoid_16bs_005lr'])

print('Model with one Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_tanh_16bs_005lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005 ',
                          'model_1hl_tanh_16bs_005lr'])

print('Model with one Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_relu_16bs_005lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ',
                          'model_1hl_relu_16bs_005lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_sigmoid_64bs_005lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ',
     'model_1hl_sigmoid_64bs_005lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_tanh_64bs_005lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005 ',
                          'model_1hl_tanh_64bs_005lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_relu_64bs_005lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ',
                          'model_1hl_relu_64bs_005lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_sigmoid_128bs_005lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ',
     'model_1hl_sigmoid_128bs_005lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_tanh_128bs_005lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ',
     'model_1hl_tanh_128bs_005lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_1hl_relu_128bs_005lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ',
     'model_1hl_relu_128bs_005lr'])

print('Model with one Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_sigmoid_16bs_02lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ',
     'model_1hl_sigmoid_16bs_02lr'])

print('Model with one Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_tanh_16bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ',
                          'model_1hl_tanh_16bs_02lr'])

print('Model with one Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_relu_16bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ',
                          'model_1hl_relu_16bs_02lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_sigmoid_64bs_02lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ',
     'model_1hl_sigmoid_64bs_02lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_tanh_64bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ',
                          'model_1hl_tanh_64bs_02lr'])

print('Model with one Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_relu_64bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ',
                          'model_1hl_relu_64bs_02lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_sigmoid_128bs_02lr')
allModelNames_1hl.append(
    ['Model with one Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ',
     'model_1hl_sigmoid_128bs_02lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_tanh_128bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02',
                          'model_1hl_tanh_128bs_02lr'])

print('Model with one Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_1, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_1hl_relu_128bs_02lr')
allModelNames_1hl.append(['Model with one Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ',
                          'model_1hl_relu_128bs_02lr'])

print('Each models Total Training Loss, Validation Loss and Validation Accuracy plots respectively \n')

allModelNames_2hl = []
print('Model with two Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_sigmoid_16bs_005lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.005 ',
     'model_2hl_sigmoid_16bs_005lr'])

print('Model with two Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_tanh_16bs_005lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.005 ',
                          'model_2hl_tanh_16bs_005lr'])

print('Model with two Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_relu_16bs_005lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.005 ',
                          'model_2hl_relu_16bs_005lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_sigmoid_64bs_005lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.005 ',
     'model_2hl_sigmoid_64bs_005lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_tanh_64bs_005lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.005 ',
                          'model_2hl_tanh_64bs_005lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_relu_64bs_005lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.005 ',
                          'model_2hl_relu_64bs_005lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_sigmoid_128bs_005lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.005 ',
     'model_2hl_sigmoid_128bs_005lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_tanh_128bs_005lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.005 ',
     'model_2hl_tanh_128bs_005lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.005)
save_model(weights, biases, 'model_2hl_relu_128bs_005lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.005 ',
     'model_2hl_relu_128bs_005lr'])

print('Model with two Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_sigmoid_16bs_02lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=16, activation function= Sigmoid, learning rate=0.02 ',
     'model_2hl_sigmoid_16bs_02lr'])

print('Model with two Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_tanh_16bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=16, activation function= tanh, learning rate=0.02 ',
                          'model_2hl_tanh_16bs_02lr'])

print('Model with two Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=16, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_relu_16bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ',
                          'model_2hl_relu_16bs_02lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_sigmoid_64bs_02lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=64, activation function= Sigmoid, learning rate=0.02 ',
     'model_2hl_sigmoid_64bs_02lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_tanh_64bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=64, activation function= tanh, learning rate=0.02 ',
                          'model_2hl_tanh_64bs_02lr'])

print('Model with two Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=64, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_relu_64bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=64, activation function= ReLU, learning rate=0.02 ',
                          'model_2hl_relu_64bs_02lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='sigmoid',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_sigmoid_128bs_02lr')
allModelNames_2hl.append(
    ['Model with two Hidden Layer, batch_size=128, activation function= Sigmoid, learning rate=0.02 ',
     'model_2hl_sigmoid_128bs_02lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='tanh',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_tanh_128bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=128, activation function= tanh, learning rate=0.02',
                          'model_2hl_tanh_128bs_02lr'])

print('Model with two Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train, y_train, x_validation, y_validation, layer_dims_2, activation_function='ReLU',
                        epochs=30, batch_size=128, learning_rate=0.02)
save_model(weights, biases, 'model_2hl_relu_128bs_02lr')
allModelNames_2hl.append(['Model with two Hidden Layer, batch_size=128, activation function= ReLU, learning rate=0.02 ',
                          'model_2hl_relu_128bs_02lr'])

print('Each models Total Training Loss, Validation Loss and Validation Accuracy plots respectively \n')

#prints all different models accuracies on Test Data
#Also , returns best model which has the highest accuracy on test data
def getModelsAccuracy(allModelNames):
    bestModelAcc= 0
    bestModelName = ''
    for eachModel in allModelNames:
        modelTitle, model = eachModel
        weights , biases = load_model(model)
        forward_results_test, predictions , labels = predict(x_test, y_test, weights, biases,
                                                             activation_function = model.split('_')[2])
        print(modelTitle)
        acc = accuracy_score(predictions,labels)*100
        if acc>bestModelAcc:
            bestModelAcc = acc
            bestModelName = model
        print("Test Data Accuracy : {:.3f}".format(acc))
    return bestModelName

#Models with No hidden layer
bestModel_0hl = getModelsAccuracy(allModelNames_0hl)

#Models with  1-hidden layer
bestModel_1hl = getModelsAccuracy(allModelNames_1hl)

#Models with 2-hidden layer
bestModel_2hl = getModelsAccuracy(allModelNames_2hl)


def anaylsis_for_model(model, x_test, y_test):
    print("""Best Model Parameters with {} Hidden Layers is:
        Batch Size : {}  , Learning Rate :0.{} , Activation Function : {} """.format(
        model.split("_")[1][:-2], model.split("_")[3][:-2], model.split("_")[4][:-2], model.split("_")[2]))

    weights, biases = load_model(model)
    forward_results_test, predictions, labels = predict(x_test, y_test, weights, biases,
                                                        activation_function=model.split('_')[2])
    cr = classification_report(predictions, labels, target_names=list(class_dict.keys()))
    print("\n Classification Report ")
    print(cr)

    cm = confusion_matrix(predictions, labels)
    plt.rc('figure', figsize=(16, 16))
    print("Confusion Matrix ")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_dict.keys())).plot(cmap="YlGn")
    print("Accuracy : {:.3f} ".format(accuracy_score(predictions, labels) * 100))


#Best Model with No Hidden Layer
anaylsis_for_model(bestModel_0hl,x_test, y_test)
#Best Model with 1-Hidden Layer
anaylsis_for_model(bestModel_1hl,x_test, y_test)
#Best Model with 2-Hidden Layer
anaylsis_for_model(bestModel_2hl,x_test, y_test)

#for plotting weight matrixes
def plot_weights(weigs, p_x, p_y, numb_of_neurons, resize_shape, output_layer=False):
    fig = plt.figure(figsize=(16, 16))
    for i in range(numb_of_neurons):
        fig.add_subplot(p_x , p_y , i+1)
        data = weigs[i]
        img = data - np.min(data);
        img = img / max(img);
        plt.axis('off')
        plt.imshow(img.reshape(resize_shape), cmap='gray')
        if output_layer:
            plt.title(reverse_class_list[i])
        else:
            plt.title('Neuron {}'.format(i+1))

#best models weights visualization
weights, biases = load_model(bestModel_0hl)
#First Layer Weights
plot_weights(weights[0], 5,5, 15 , (60,60), True)

#1st hidden layer weights
weights, biases = load_model(bestModel_1hl)
plot_weights(weights[0], 12,12, 144 , (60,60), False)

#output layer weights
plot_weights(weights[1], 5,5, 15 , (12,12), True)
weights, biases = load_model(bestModel_2hl)

#1st hidden layer weights
plot_weights(weights[0], 12,12, 144 , (60,60), False)
#2nd hidden layer weights
plot_weights(weights[1], 8,8, 64 , (12,12), False)
#output layer weights
plot_weights(weights[2], 5,5, 15 , (8,8), True)

def printForHtmlTable(allModelNames):
    for eachModel in allModelNames:
        modelTitle, model = eachModel
        weights , biases = load_model(model)
        forward_results_test, predictions , labels = predict(x_test, y_test, weights, biases,
                                                            activation_function = model.split('_')[2])
        acc = accuracy_score(predictions,labels)*100
        print('<td style="border: 1px solid black;" colspan="2"> {:.3f} </td>'.format(acc))
# printForHtmlTable(allModelNames_0hl)
# print()
# printForHtmlTable(allModelNames_1hl)
# print()
# printForHtmlTable(allModelNames_2hl)

# reading our train , validation and test data with rgb values
x_train_rgb, y_train_rgb = read_images(trainpath, class_dict, size=(60, 60), convert_gray=False)
x_validation_rgb, y_validation_rgb = read_images(validationpath, class_dict, size=(60, 60), convert_gray=False)
x_test_rgb, y_test_rgb = read_images(testpath, class_dict, size=(60, 60), convert_gray=False)

# shuffling train data
keys = np.array(range(x_train_rgb.shape[1]))
np.random.shuffle(keys)
x_train_rgb = x_train_rgb[:, keys]
y_train_rgb = y_train_rgb[:, keys]

# visualizing some random RGB-images from our data
fig = plt.figure(figsize=(16, 16))
for i in range(1, 37):
    img_index = np.random.randint(0, x_train_rgb.shape[1])
    fig.add_subplot(6, 6, i)
    plt.imshow(x_train_rgb[:, img_index].reshape((60, 60, 3)))
    plt.axis('off')
    plt.title(reverse_class_list[np.argmax(y_train_rgb[:, img_index], axis=0)])

#Using Best Parameters for 0 Hidden Layer Model

# Best Model Parameters with 0 Hidden Layers is:
#         Batch Size : 16  , Learning Rate :0.005 , Activation Function : sigmoid

layer_dims_0_rgb = [input_layer*3 , output_layer ]  # No hidden layers
layer_dims_1_rgb = [input_layer*3, 144 , output_layer ]  # 1 hidden layer
layer_dims_2_rgb = [input_layer*3, 144 , 64 , output_layer ]  # 2 hidden layers

print('Model with No Hidden Layer, batch_size=16, activation function= sigmoid, learning rate=0.005 ')
weights, biases = train(x_train_rgb, y_train_rgb, x_validation_rgb, y_validation_rgb, layer_dims_0_rgb,  activation_function='sigmoid',
                        epochs = 30, batch_size = 16, learning_rate=0.005)

save_model(weights, biases, 'model_0hl_sigmoid_16bs_005lr_rgb')

anaylsis_for_model('model_0hl_sigmoid_16bs_005lr_rgb',x_test_rgb, y_test_rgb)

#weights of 1st hidden layer which is output layer
weights, biases =load_model('model_0hl_sigmoid_16bs_005lr_rgb')
plot_weights(weights[0], 5,5, 15 , (60,60,3), True)

#Using Best Parameters for 1 Hidden Layer Model

# Best Model Parameters with 1 Hidden Layers is:
#         Batch Size : 16  , Learning Rate :0.02 , Activation Function : relu

print('Model with One Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train_rgb, y_train_rgb, x_validation_rgb, y_validation_rgb, layer_dims_1_rgb,
                        activation_function='ReLU', epochs = 30, batch_size = 16, learning_rate=0.02)

save_model(weights, biases, 'model_1hl_relu_16bs_02lr_rgb')
anaylsis_for_model('model_1hl_relu_16bs_02lr_rgb',x_test_rgb, y_test_rgb)

#weights of 1st hidden layer
weights, biases =load_model('model_1hl_relu_16bs_02lr_rgb')
plot_weights(weights[0], 12,12, 144 , (60,60,3), False)

#Using Best Parameters for 0 Hidden Layer Model

# Best Model Parameters with 2 Hidden Layers is:
#         Batch Size : 16  , Learning Rate :0.02 , Activation Function : relu

np.seterr(divide = 'ignore')
print('Model with Two Hidden Layer, batch_size=16, activation function= ReLU, learning rate=0.02 ')
weights, biases = train(x_train_rgb, y_train_rgb, x_validation_rgb, y_validation_rgb, layer_dims_2_rgb,
                        activation_function='ReLU', epochs = 30, batch_size = 16, learning_rate=0.02)

save_model(weights, biases, 'model_2hl_relu_16bs_02lr_rgb')

anaylsis_for_model('model_2hl_relu_16bs_02lr_rgb',x_test_rgb, y_test_rgb)

#weights of 1st hidden layer
weights, biases =load_model('model_2hl_relu_16bs_02lr_rgb')
plot_weights(weights[0], 12,12, 144 , (60,60,3), False)


# Neccessary libraries for Part-2
import torch
from torch import optim
import torch.nn as nn
import torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision import models
import torchvision.transforms as transforms
import seaborn as sn
import pandas as pd
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import  DataLoader
from torch.utils.data.dataloader import default_collate

train_path = "Vegetable Images/train"                  # path for training data
validation_path = "Vegetable Images/validation"        # path for validation data
test_path = "Vegetable Images/test"                    # path for test data

classes = {}
l1 = os.listdir(train_path)
l1.sort()

for i, folder in enumerate(l1):
    classes[folder] = i

classes  # 15 classes which are labels for the output layer

def read_images_CNN(path, classes):                       # read all images in the given path
    l = []
    for folder in os.listdir(path):
        for file_path in glob.glob(str(path +'/' + folder + '/*.jpg')):
            img = Image.open(file_path)
            img = img.resize((60,60))                     # resize all images to 32x32
            tmp = (img,classes[folder])
            l.append(tmp)
    return l

train_set = read_images_CNN(train_path, classes)                   # read train images
validation_set = read_images_CNN(validation_path, classes)         # read validation images
test_set = read_images_CNN(test_path, classes)                     # read test images

# first element of a test set which is a tuple with RGB image sized 32x32 and its class label
img = test_set[0]
img

# same classes within a tuple indexed 0 to 15
CNN_classes = ('Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
               'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato')

my_CNN_classes = list(CNN_classes)


def convert_to_tensor(dataset):  #  convert the given dataset to pytorch tensor
    for i in range(len(dataset)):
        convert_tensor = transforms.ToTensor()
        tmp = (convert_tensor(dataset[i][0]), dataset[i][1])
        dataset[i] = tmp


convert_to_tensor(train_set)  # train set converting
convert_to_tensor(validation_set)  # validation set converting
convert_to_tensor(test_set)  # test set converting


# all tensors in train, validation, test sets will run on GPU
def do_cuda(dataset):
    for i in range(len(dataset)):
        a, b = dataset[i][0].to("cuda"), dataset[i][1]
        dataset[i] = (a, b)


do_cuda(train_set)
do_cuda(validation_set)
do_cuda(test_set)


# get the all datasets' loaders
def get_loaders(batch_size, train_set, validation_set, test_set):
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)

    return train_loader, validation_loader, test_loader

train_loader, validation_loader, test_loader = get_loaders(32, train_set, validation_set, test_set)

# train and test transforms
train_transform = transforms.Compose([
    transforms.Resize((60, 60)),
    # transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# run all the code in GPU(cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# VGG19 CNN model from pytorch library
model = models.vgg19(pretrained = True)
model2 = models.vgg19(pretrained = True)

# disable model's parameters
def grad_parameters(mdl):
    for p in mdl.parameters():
        p.requires_grad = False


grad_parameters(model)
grad_parameters(model2)

model.classifier = nn.Sequential(

    nn.Linear(in_features=25088, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=0.6),

    nn.Linear(in_features=512, out_features=15),
    nn.LogSoftmax(dim=1)
)

model2.classifier = nn.Sequential(

    nn.Linear(in_features=25088, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=0.6),

    nn.Linear(in_features=512, out_features=15),
    nn.LogSoftmax(dim=1)
)

# model parameters
model

# two different model:
# one of the finetunes all of its layers
# the other one just finutes just last two fully connected layers
model = model.to(device)
model2 = model2.to(device)


# optimizer for first model
# all parameters will be updated
optimizer = optim.Adam(model.parameters(), lr = 0.005)
criterion = nn.CrossEntropyLoss()

# just last two fully connected layers to update
params_to_update = []
model2_parameters = list(model2.parameters())
params_to_update = model2_parameters[-2:]
# optimizer for second model
optimizer2 = optim.SGD(params_to_update, lr = 0.005)
criterion2 = nn.CrossEntropyLoss()


# get all the predictions for the entire training set
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).to(device)
    model.eval()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


# return the accuracy of the model
def get_accuracy(model, loader):
    with torch.no_grad():  # disable gradient computations
        test_preds = get_all_preds(model, loader)

    l = []
    for i in range(len(test_set)):
        l.append(test_set[i][1])

    l = np.array(l)

    acc_score = accuracy_score(l, test_preds.cpu().argmax(dim=1).numpy())

    return acc_score


def CNN_train(train_loader, validation_loader, model, criterion, optimizer, device, num_epochs):
    acc_list = []
    last_acc = 0  # will be used for early-stopping
    t = time.time()
    for epoch in range(num_epochs):
        # Load in the data in batches using the train_loader object
        for images, labels in train_loader:
            # Move tensors to the configured device

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_score = get_accuracy(model, validation_loader)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            print('Epoch {}, Loss: {:.4f}'.format(epoch + 1, loss.item()))
            # acc_score = get_accuracy(model, validation_loader)
            print("Validation accuracy:", acc_score)

            #  if new calculated accuracy is lower than 10 epochs before accuracy exit the function
            if acc_score < last_acc:
                print("Early stopping applied.")
                print("Passed time for training in seconds is:", time.time() - t)
                return acc_list

            last_acc = acc_score

        acc_list.append(acc_score)

    print("Passed time for training in seconds is:", time.time() - t)
    return acc_list

# first model training process
acc_list_1 = CNN_train(train_loader, validation_loader, model, criterion, optimizer, device, num_epochs=35)
# second model training process
acc_list_2 = CNN_train(train_loader, validation_loader, model2, criterion2, optimizer2, device, num_epochs=35)


# testing the CNN model
def CNN_test(model, loader):
    with torch.no_grad():  # disable gradient computations
        test_preds = get_all_preds(model, loader)

    l = []
    for i in range(len(test_set)):
        l.append(test_set[i][1])

    l = np.array(l)

    cr = classification_report(l, test_preds.cpu().argmax(dim=1).numpy(), labels=np.unique(l))
    cf = confusion_matrix(l, test_preds.cpu().argmax(dim=1).numpy(), labels=np.unique(l))
    return cr, cf

cr1, cf1 = CNN_test(model, test_loader)
cr2, cf2 = CNN_test(model2, test_loader)

# early-stopping has occured in the first model, and did not train for last 10 epochs
plt.plot(list(range(0, len(acc_list_1))), acc_list_1)
plt.ylabel('Validation accuracy for the first model')
plt.show()

plt.plot(list(range(0, len(acc_list_2))), acc_list_2)
plt.ylabel('Validation accuracy for the second model')
plt.show()

# for displaying the confusion matrix
def show_confusion_matrix(cf, CNN_classes):
    df_cm = pd.DataFrame(cf, index = [i for i in CNN_classes],
              columns = [i for i in CNN_classes])
    plt.figure(figsize = (15,15))
    sn.heatmap(df_cm, annot=True,cmap="YlGn")


# first model whose all parameters has been trained
acc1 = get_accuracy(model, test_loader)
print(cr1)
print("Accuracy:", acc1 * 100)

show_confusion_matrix(cf1, CNN_classes)

# second model whose just last 2 fully connected layers has been trained
acc2 = get_accuracy(model2, test_loader)
print(cr2)
print("Accuracy:", acc2 * 100)

show_confusion_matrix(cf2, CNN_classes)


# funciton to visualize the CNN layer image presentations
def custom_viz(kernels, path=None, cols=None):
    def set_size(w, h, ax=None):
        if not ax:
            ax = plt.gca()

        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - l)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    N = kernels.shape[0]
    C = kernels.shape[1]

    Tot = N * C

    if C > 1:
        columns = C
    elif cols == None:
        columns = N
    elif cols:
        columns = cols

    rows = Tot // columns
    rows += Tot % columns

    pos = range(1, Tot + 1)

    fig = plt.figure(1)
    fig.tight_layout()
    k = 0

    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(rows, columns, pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k + 1

    set_size(30, 30, ax)

    """if path:
        plt.savefig(path, dpi=100)
    """
    plt.show()

# an example image
img[0]

# first layer's presentation
kernels = model.features[0].weight.cpu().detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
custom_viz(kernels, img[0], "layer1")

# next layer's presentation
kernels = model.features[2].weight.cpu().detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
custom_viz(kernels, img[0], "layer2")

# next layer's presentation
kernels = model.features[5].weight.cpu().detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
custom_viz(kernels, img[0], "layer3")