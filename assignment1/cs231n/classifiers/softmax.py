import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(X.shape[0]):
    scores = W.T.dot(X[i,:])
    scores -= scores.max()
    scores = np.exp(scores)/(np.exp(scores).sum())
    loss -= np.log(scores[y[i]])

    one_hot = np.zeros_like(scores)
    one_hot[y[i]] = 1
    dW += np.outer(X[i,:], scores - one_hot)

  loss /= X.shape[0]
  loss += reg*(W*W).sum() # add regularization loss

  dW /= X.shape[0]
  dW += 2*reg*W # add regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = W.T.dot(X.T)
  scores -= scores.max(axis=0)
  scores = np.exp(scores)/(np.exp(scores).sum(axis=0))

  one_hot_y = np.eye(W.shape[1])[y] # number of classes = W.shape[1]
  loss = - (1./X.shape[0]) * (np.log(scores) * one_hot_y.T).sum() + reg*(W*W).sum()
  dW = (1./X.shape[0]) * X.T.dot((scores - one_hot_y.T).T) + 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

