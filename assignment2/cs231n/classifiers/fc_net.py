from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim*hidden_dim).reshape((input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim*num_classes).reshape((hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Begin forward pass (but not including softmax operation)
        out_affine_1, cache_affine_1 = affine_forward(X, self.params['W1'], self.params['b1'])
        out_relu, cache_relu = relu_forward(out_affine_1)        
        out_affine_2, cache_affine_2 = affine_forward(out_relu, self.params['W2'], self.params['b2'])
        scores = out_affine_2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Complete forward pass
        loss, dout = softmax_loss(scores, y)
        # Perform backward pass
        dout, grads['W2'], grads['b2'] = affine_backward(dout, cache_affine_2)
        dout = relu_backward(dout, cache_relu)
        dout, grads['W1'], grads['b1'] = affine_backward(dout, cache_affine_1)

        # Include regularization impact on loss and gradients
        loss += (0.5 * self.reg * np.square(self.params['W1']).sum() + 
                 0.5 * self.reg * np.square(self.params['W2']).sum())
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        # Initialize affine layer parameters (i.e. weights and biases)
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, self.num_layers+1): # have indexing start at 1
            weight_str = '{}{}'.format('W', i)
            bias_str = '{}{}'.format('b', i)
            gamma_str = '{}{}'.format('gamma', i)            
            beta_str = '{}{}'.format('beta', i)
            input_dim = layer_dims[i-1]
            output_dim = layer_dims[i]
            self.params[weight_str] = weight_scale * np.random.randn(input_dim,output_dim)
            self.params[bias_str] = np.zeros(output_dim)
            if self.normalization is not None and i < self.num_layers: 
                # Don't perform normalization on last layer
                # Note: these parameters are used for both batch and layer normalization
                self.params[gamma_str] = np.ones(output_dim)
                self.params[beta_str] = np.zeros(output_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # PERFORM FORWARD PASS
        # --------------------------------------------------------------------
        # Must proceed in correct order, i.e.
        #       affine --> [batch/layer norm] --> relu --> [dropout]
        # Note: Only computes scores for each class; doesn't perform softmax
        #       operation; softmax takes place below when computing loss
        # --------------------------------------------------------------------

        # Create dictionaries to store cache information
        affine_cache_dict = dict()
        relu_cache_dict = dict()
        norm_cache_dict = dict()
        dropout_cache_dict = dict()
        out = X # This is first input

        # Loop through layers, completing a forward pass for each one
        for i in range(1,self.num_layers+1):
            weight_str = '{}{}'.format('W', i)
            bias_str = '{}{}'.format('b', i)
            gamma_str = '{}{}'.format('gamma', i)
            beta_str = '{}{}'.format('beta', i)

            # FORWARD PASS: Affine
            out, cache_affine = affine_forward(out,
                                               self.params[weight_str], 
                                               self.params[bias_str])
            affine_cache_dict[i] = cache_affine
            if i == self.num_layers:
                # On the final layer, only perform the affine forward pass; break so 
                # we don't perform perform ReLU, batch/layer norm, or dropout passes
                break

            # FORWARD PASS: batch/layer norm
            if self.normalization is not None:
                if self.normalization is 'batchnorm':
                    norm_function_forward = batchnorm_forward
                elif self.normalization is 'layernorm':
                    norm_function_forward = layernorm_forward
                else:
                    raise ValueError('Invalid normalization type')
                out, cache_norm = norm_function_forward(out, 
                                                        self.params[gamma_str], 
                                                        self.params[beta_str], 
                                                        self.bn_params[i-1])
                norm_cache_dict[i] = cache_norm

            # FORWARD PASS: ReLU
            out, cache_relu = relu_forward(out)
            relu_cache_dict[i] = cache_relu

            # FORWARD PASS: dropout
            if self.use_dropout:
                out, cache_dropout = dropout_forward(out, self.dropout_param)
                dropout_cache_dict[i] = cache_dropout

        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Complete forward pass
        loss, dout = softmax_loss(scores, y)

        # PERFORM BACKWARD PASS
        # --------------------------------------------------------------------
        # Must proceed in correct order, i.e.
        #       [dropout] --> ReLU --> [batch/layer norm] --> affine
        # --------------------------------------------------------------------

        # Loop through layers in reverse, starting at self.num_layers, ending at 1
        for i in range(self.num_layers, 0, -1):
            weight_str = '{}{}'.format('W', i)
            bias_str = '{}{}'.format('b', i)
            gamma_str = '{}{}'.format('gamma', i)
            beta_str = '{}{}'.format('beta', i)

            if i < self.num_layers:
                # On the final layer, only perform the affine backward 
                # pass. So, only perform ReLU, batch/layer norm, or 
                # dropout backward passes when we're NOT on the final layer

            # BACKWARD PASS: dropout
                if self.use_dropout:
                    dout = dropout_backward(dout, dropout_cache_dict[i])

            # BACKWARD PASS: ReLU
                dout = relu_backward(dout, relu_cache_dict[i])

            # BACKWARD PASS: batch/layer norm
                if self.normalization is not None:
                    if self.normalization is 'batchnorm':
                        norm_function_backward = batchnorm_backward_alt
                    elif self.normalization is 'layernorm':
                        norm_function_backward = layernorm_backward
                    else:
                        raise ValueError('Invalid normalization type')
                    dout, grads[gamma_str], grads[beta_str] = norm_function_backward(
                                                                dout,
                                                                norm_cache_dict[i])

            # BACKWARD PASS: Affine
            dout, grads[weight_str], grads[bias_str] = affine_backward(
                                                            dout, 
                                                            affine_cache_dict[i])

            # Account for regularization in gradient
            grads[weight_str] += self.reg * self.params[weight_str]

            # Account for regularization in loss
            loss += 0.5 * self.reg * np.square(self.params[weight_str]).sum()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads






