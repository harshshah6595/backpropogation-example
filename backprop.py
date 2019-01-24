import gzip
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def loading(doc, n,type_file):
    
    if type_file==1:
        
        with gzip.open(doc) as d:
            d.read(16)
            mb = d.read(28 * 28 * n)
            output = np.frombuffer(mb,np.uint8).astype(np.float32)
            output = output.reshape(n, 28, 28)
        return output
    
    else:
        with gzip.open(doc) as d:
            d.read(8)
            mb = d.read(1 * n)
            output = np.frombuffer(mb,np.uint8).astype(np.int64)
        return output
    
train_X = loading('train-images-idx3-ubyte.gz', 60000,1)

test_X = loading('t10k-images-idx3-ubyte.gz', 10000,1)

trY = loading('train-labels-idx1-ubyte.gz', 60000,2)

teY = loading('t10k-labels-idx1-ubyte.gz', 10000,2)
plt.imshow(train_X[1])
trX=train_X.reshape(60000,784)
teX=test_X.reshape(10000,784)


trX = trX/255
nn_input_dim = 784 # input layer dimensionality
nn_output_dim = 10 # output layer dimensionality
train_errors=[]
test_errors=[]
epsilon = 0.001 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(100), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./100 * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=30, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(110)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    energy_list=[] 
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        for j in range(0,60000,100):
            X = trX[j:j+100]
            y = trY[j:j+100]

            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            dy2 = probs
            dy2[range(100), y] -= 1
            dW2 = (a1.T).dot(dy2)
            db2 = np.sum(dy2, axis=0, keepdims=True)
            dy1 = dy2.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, dy1)
            db1 = np.sum(dy1, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2yo
            b2 += -epsilon * db2

            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y)))
            energy_list.append(calculate_loss(model, X, y))
            z_pred=predict(model,trX)
            y_pred = predict(model, teX)
            accuracy = np.mean(y_pred == teY)
            sum_te_misc=np.sum(y_pred==teY)
            sum_tr_misc=np.sum(z_pred==trY)
            train_errors.append(60000-sum_tr_misc)
            test_errors.append(10000-sum_te_misc)

            print("Accuracy after iteration %i: %f" %(i, accuracy))
    plt.title("loss")
    plt.plot(range(len(energy_list)), energy_list)
    plt.show()
    plt.title("train misclassification")
    plt.plot(range(len(train_errors)), train_errors)
    plt.show()
    plt.title("test misclassification")
    plt.plot(range(len(test_errors)), test_errors)
    plt.show()
    return model
    

# Build a model with a 400-dimensional hidden layer
model = build_model(400, print_loss=True)
