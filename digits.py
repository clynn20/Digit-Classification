import numpy as np
import matplotlib.pyplot as plt
import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib
font = {'weight' : 'normal',
        'size' : 22}
matplotlib.rc('font', **font)

# global parameters for stochastic gradient descent
np.random.seed(102)
step_size = 0.01
batch_size = 200
max_epochs = 200

# global parameters for network
num_of_layers = 2
width_of_layers = 20
activation = "ReLU" if False else "Sigmoid"


class LinearLayer:
    # initialize the layer with input_dim x output_dim weight matrix and a bias vector 1 x output_dim) 
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float64) * np.sqrt(2. / input_dim)
        self.bias = np.ones((1, output_dim))*0.5
        
    # compute Xw+b during forward pass
    def forward(self, input):
        self.input = input
        return self.input@self.weights + self.bias
    
    # backward pass 
    # grad -- dL/dZ, for a batch of size n, grad is a n x output_dim matrix where the ith row is the gradient of the loss 
    # of example i with respect to z_i
    # self.grad_weighs -- dL/dW, a input_dim x output_dim matrix storing the gradient of the loss with respect to the weight of this layer
    # self.grad_bias -- dL/db, a 1 x output_dim matrix storing the gradient of hte loss with respect to the bias of this layer
    # grad_input -- dL/dX, for a batch size of n, grad_input is a n x input_dim matrix where the ith row is the gradient of the loss of 
    # example i with respect to x_i
    def backward(self, grad):
        self.grad_weights = self.input.T@grad
        self.grad_bias = np.sum(grad, axis=0)
        grad_input = grad@self.weights.T
        return grad_input
    
    # update the weights and biases based on the stored gradients form the backward pass
    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias

# neural network
class FeedForwardNeuralNetwork:
    # build a network of linear layers separated by non linear activations either Sigmoid or ReLU
    # each internal layer has hidden_dim dimension
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation="ReLU"):
        # just a linear mapping from input to output
        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        # at least two layers
        else:
            # layer to map input to hidden dimension size
            self.layers = [LinearLayer(input_dim, hidden_dim)]
            self.layers.append(Sigmoid() if activation=="Sigmoid" else ReLU())
            
            # hidden layers
            for _ in range(num_layers-2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(Sigmoid() if activation=="Sigmoid" else ReLU())
            # layer to map hidden dimension to output size
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    # given an input, call the forward function of each of the layers
    # pass the output of each layer to the next one
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    # given an gradient with respect to the network output, call the backward function of each of the layers.
    # pass the output of each layer to the one before it.
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return
        
    # tell each layer to update its weight based on the gradient computed in the backward pass
    def step(self, step_size=0.001):
        for layer in self.layers:
            layer.step(step_size)
        return

# sigmoid activation function
class Sigmoid:
    # given the input value, apply the sigmoid function, and store the output value for use in backward pass
    def forward(self,input):
        self.act = 1/(1+np.exp(-input))
        return self.act
    # compute the gradient of the output with respect to the input self.act*(1-self.act)
    # and then multiply by the loss gradient with respect to the output produce the loss 
    # gradient with respect to the input
    def backward(self, grad):
        return grad * self.act * (1-self.act)
    # the sigmoid function has no parameter so do nothing
    def step(self, step_size):
        return   
    
# ReLU activation function
class ReLU:
    # forward pass is max(0,input)
    def forward(self, input):
        self.mask = (input>0)
        return input.self.mask
    # backward pass masks out same elements
    def backward(self, grad):
        return grad * self.mask
    # the ReLU function has no parameter so do nothing
    def step(self, step_size):
        return

class CrossEntropySoftmax:
    def forward(self, logits, labels):
        self.probs = softmax(logits)
        self.labels = labels
        return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:, np.newaxis], labels] + 0.00001))
    
    def backward(self):
        grad = self.probs
        grad[np.arange(len(self.probs))[:, np.newaxis], self.labels] -= 1
        return grad.astype(np.float64) / len(self.probs)

# compute the loss 
def softmax(x):
    x -=np.max(x, axis=1)[:, np.newaxis]
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, np.newaxis])

# validation
def evaluateValidation(model, x_val, y_val, batch_size):
    val_loss_running = 0
    val_acc_running = 0
    j = 0
    lossFunc = CrossEntropySoftmax()
    while j < len(x_val):
        b = min(batch_size, len(x_val)-j)
        x_batch = x_val[j:j+b]
        y_batch = y_val[j:j+b].astype(np.int64)
        logits = model.forward(x_batch)
        loss = lossFunc.forward(logits, y_batch)
        acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == y_batch)
        val_loss_running += loss*b
        val_acc_running += acc*b
        j += batch_size
    return val_loss_running/len(x_val), val_acc_running/len(x_val)

# load data 
def loadData(normalize = True):
    train = np.loadtxt("mnist_small_train.csv", delimiter=",", dtype=np.float64)
    val = np.loadtxt("mnist_small_val.csv", delimiter=",", dtype=np.float64)
    test = np.loadtxt("mnist_small_test.csv", delimiter=",", dtype=np.float64)
    
    # normalize the data
    if normalize:
        x_train = train[:,:-1]/256-0.5
        x_val = val[:,:-1]/256-0.5
        x_test = test/256-0.5
    else:
        x_train = train[:,:-1]
        x_val = val[:,:-1]
        x_test = test
    y_train = train[:,-1].astype(np.int64)[:, np.newaxis]
    y_val = val[:,-1].astype(np.int64)[:, np.newaxis]
    logging.info("Loaded train " + str(x_train.shape))
    logging.info("Loaded val: " + str(x_val.shape))
    logging.info("Loaded test: " + str(x_test.shape))
    return x_train, y_train, x_val, y_val, x_test

# display a single image
def show_img(images_set, idx):
    plt.figure()
    plt.imshow(images_set[idx].reshape(28,28), cmap="gray")
    plt.colorbar()
    plt.grid(False)
    plt.show()

def main():
    # load data and display
    x_train, y_train, x_val, y_val, x_test = loadData()
    
    # display a single image 
    #show_img(x_test, 0)
    
    # build a network with input feature dimensions, output feature dimensions,
    # hidden dimension, and number of layers as specified below
    net = FeedForwardNeuralNetwork(x_train.shape[1], 10, width_of_layers, num_of_layers, activation=activation)
    
    losses = []
    val_losses = []
    accs = []
    val_accs = []
    
    # loss function
    lossFunc = CrossEntropySoftmax()
    
    # indices we will use to shuffle data randomly
    inds = np.arange(len(x_train))
    for i in range(max_epochs):
        # shuffled indices so we go through data in new random batches
        np.random.shuffle(inds)
        # go through all data points once 
        j = 0
        acc_running = loss_running = 0
        while j < len(x_train):
            b = min(batch_size, len(x_train)-j)
            x_batch = x_train[inds[j:j+b]]
            y_batch = y_train[inds[j:j+b]].astype(np.int64)
            
            # compute the scores for the 10 classes
            logits = net.forward(x_batch)
            loss = lossFunc.forward(logits, y_batch)
            acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == y_batch)
            
            # compute gradient of cross entropy loss with respect to logits
            loss_grad = lossFunc.backward()
            
            # pass gradient back through networks
            net.backward(loss_grad)
            
            # take a step of gradient descent
            net.step(step_size)
            
            # record losses and accuracy then move to next batch
            losses.append(loss)
            accs.append(acc)
            loss_running += loss*b
            acc_running += acc*b
            j += batch_size
        
        # evaluate performance on validation
        vloss, vacc = evaluateValidation(net, x_val, y_val, batch_size)
        val_losses.append(vloss)
        val_accs.append(vacc)
        
        # print out the average stats over each epoch
        logging.info(f"[Epoch {i:3}]    Loss: {loss_running/len(x_train):8.4}    Train Acc: {acc_running/len(x_train)*100:8.4}%     Val Acc: {vacc*100:8.4}%")

    # run prediction on test set
    predict_digits = np.argmax(net.forward(x_test), axis=1)[:, np.newaxis]
    id_n_digits = np.concatenate((np.expand_dims(np.array(range(len(x_test)), dtype=np.int64), axis=1), predict_digits), axis=1)
    header = np.array([["id", "digit"]])
    txt_out = np.concatenate((header, id_n_digits))
    np.savetxt("test_predicted.csv", txt_out, fmt='%s', delimiter=",")
        
    # plot the performance
    fig, ax1 = plt.subplots(figsize=(16,9))
    color = "tab:red"
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i+1)*len(x_train)/batch_size) for i in range(len(val_losses))], val_losses, c="red", label="Validation Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01,3)
        
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i+1)*len(x_train)/batch_size) for i in range(len(val_accs))], val_accs, c="blue", label="Validation Acc.")
    ax2.set_ylabel("Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01,1.01)
    fig.tight_layout()
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()
        
if __name__=="__main__":
    main()