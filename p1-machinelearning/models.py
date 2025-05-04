from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        # Let's initialize the weight as a PyTorch parameter
        self.w = Parameter(torch.randn(1,dimensions)) # We initialie randomly

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return torch.matmul(x, self.w.T) # We compute the dot product of input and weights

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        if score >= 0:
            return torch.tensor(1)
        else:
            return torch.tensor(-1)


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # We iterate over the batches
        correct = False
        
        while not correct:
            correct = True # We assume the classification is correct until proven otherwise
            for batch in dataloader:
                x, label = batch['x'], batch['label']
                prediction = self.get_prediction(x)
                misclassified = (prediction != label) # We check where the missclassification occurs
                if misclassified.any() :
                    correct = False # We found a misclassification, we have to continue the training
                    for i in range(len(misclassified)):
                        if misclassified[i]:
                            # We update the weights with the learning rule: w += learning_rate * x * label
                            self.w.data += 1 * x[i] * label[i] # We update weights for missclassified points ## A learning rate of 0.1 or 0.5 is too small
            


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super(RegressionModel, self).__init__()
        
        # We begin be defining the network layers.
        self.hidden1 = Linear(1, 64)  # Input to first hidden layer
        self.hidden2 = Linear(64, 64)  # First hidden layer to second hidden layer
        self.output = Linear(64, 1)  # Final output layer

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.hidden1(x))  # Apply first hidden layer + ReLU activation
        x = relu(self.hidden2(x))  # Apply second hidden layer + ReLU activation
        return self.output(x)  # Output layer
        
        
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predictions = self(x)  # Get the model's predictions
        return mse_loss(predictions, y)  # Use MSE loss
        

    def train(self, dataset, num_epochs=10000, lr=0.0001):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset : a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                x, y = batch['x'], batch['label']  # Get inputs and labels from the dataset
                
                optimizer.zero_grad()  # Zero the gradients
                loss = self.get_loss(x, y)  # Calculate loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update model parameters

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

            # Early stopping if loss is sufficiently small
            if total_loss / len(dataloader) < 0.01:
                print(f'Early stopping at epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
                break
        







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super(DigitClassificationModel, self).__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"

        self.hidden1 = Linear(input_size, 256)
        self.hidden2 = Linear(256, 256)
        self.output = Linear(256, output_size)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = relu(self.hidden1(x))
        x = relu(self.hidden2(x))
        return self.output(x)
 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predictions = self.run(x)
        return cross_entropy(predictions, y)
    
        

    def train(self, dataset, num_epochs = 15, lr = 0.001, early_stopping_threshold=0.975):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                x, y = batch['x'], batch['label']  # Get inputs and labels from the dataset
                
                optimizer.zero_grad()  # Zero the gradients
                loss = self.get_loss(x, y)  # Calculate loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update model parameters

                total_loss += loss.item()

            # Compute validation accuracy
            val_accuracy = dataset.get_validation_accuracy()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}, Validation Accuracy: {val_accuracy}')

            # Early stopping based on validation accuracy
            if val_accuracy > early_stopping_threshold:
                print(f'Early stopping at epoch {epoch+1}, Validation Accuracy: {val_accuracy}')
                break




class LanguageIDModel(Module):
    
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        # Initialize parameters
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.num_languages = len(self.languages)
        self.hidden_size = 256  # Size of the hidden state
        
        super(LanguageIDModel, self).__init__()
        
        # Initialize layers
        self.W_x = Linear(self.num_chars, self.hidden_size)  # Input to hidden weight matrix
        self.W_hidden = Linear(self.hidden_size, self.hidden_size)  # Hidden to hidden weight matrix
        self.output_layer = Linear(self.hidden_size, self.num_languages)  # Output layer for classification

        

    def run(self, xs):
        
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        
        "*** YOUR CODE HERE ***"
        h = torch.zeros(xs[0].size(0), self.hidden_size)  # Initialize hidden state (batch_size, hidden_size)
        
        for i in range(len(xs)):
            if i == 0:
                h = relu(self.W_x(xs[i]))  # Process first character
            else:
                h = relu(self.W_x(xs[i]) + self.W_hidden(h))  # Update hidden state with subsequent characters
        
        output = self.output_layer(h) 
        return output
        
    
    def get_loss(self, xs, y):
        
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        
        "*** YOUR CODE HERE ***"
        predictions = self.run(xs)  # Get language prediction scores
        y = torch.argmax(y, dim=1)  # Convert one-hot to class indices
        return cross_entropy(predictions, y)  # Cross-entropy loss for classification
                

    def train(self, dataset, num_epochs=20, lr=0.001, early_stopping_threshold=0.85):
        
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        
        "*** YOUR CODE HERE ***"
        
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                xs, y = batch['x'], batch['label']  # Get the input (xs) and labels (y)
                xs = movedim(xs, 0, 1)
                    
                optimizer.zero_grad()  # Zero the gradients
                loss = self.get_loss(xs, y)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                total_loss += loss.item()


            # Compute validation accuracy
            val_accuracy = dataset.get_validation_accuracy()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}, Validation Accuracy: {val_accuracy}')

            # Early stopping based on validation accuracy
            if val_accuracy > early_stopping_threshold:
                print(f'Early stopping at epoch {epoch+1}, Validation Accuracy: {val_accuracy}')
                break
  

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    d1, d2 = input.shape
    h, w = weight.shape

    output_d1 = d1 - h + 1
    output_d2 = d2 - w + 1

    output_tensor = torch.zeros(output_d1, output_d2)

    # Apply convolution by sliding the kernel over the input
    for i in range(output_d1):
        for j in range(output_d2):
            # Get the region of the input matrix we are currently looking at
            region = input[i:i+h, j:j+w]
            # Perform element-wise multiplication and sum to get the result
            output_tensor[i, j] = torch.sum(region * weight)
    
    return output_tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize the model parameters
        super(DigitConvolutionalModel, self).__init__()
        
        # Define the convolutional weights (3x3 filter)
        self.convolution_weights = Parameter(torch.randn(3, 3))  # Randomly initialized weights for the convolution
        
        # Fully connected layers after convolution
        self.fc1 = Linear(26 * 26, 128)  # Flattened size after convolution (26x26 from 28x28 - 3x3 filter)
        self.fc2 = Linear(128, 10)  # Output size: 10 (for MNIST digits)



    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. 
        You should treat x as a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        batch_size = x.size(0)  # Number of samples in the batch

        # Reshape input to be (batch_size, 28, 28)
        x = x.reshape(batch_size, 28, 28)
        
        # Apply convolution to each sample in the batch (use Convolve function)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))

        # Flatten the convolution output (26x26 -> 1D vector of size 676)
        x = x.flatten(start_dim=1)

        # Pass through the fully connected layers
        x = relu(self.fc1(x))  # Apply first linear layer + ReLU activation
        x = self.fc2(x)  # Output layer (logits)

        return x  # Return the final output logits

        
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """  
        # Get predictions (logits)
        predictions = self.forward(x)

        # Cross-entropy loss for classification
        return cross_entropy(predictions, y)
        

    def train(self, dataset, num_epochs=10, lr=0.001, early_stopping_threshold=0.85):
        """
        Trains the model.
        """
        # Use the Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True) # We iterate over the batches
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                xs, y = batch['x'], batch['label']  # Get the input (xs) and labels (y)

                optimizer.zero_grad()  # Zero the gradients
                loss = self.get_loss(xs, y)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                total_loss += loss.item()

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataset)}')

            # Compute validation accuracy
            val_accuracy = dataset.get_validation_accuracy()
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy}')

            # Early stopping based on validation accuracy
            if val_accuracy > early_stopping_threshold:
                print(f'Early stopping at epoch {epoch+1}, Validation Accuracy: {val_accuracy}')
                break



class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""

     