
## About The Project
This project is an assignment
 from the course "Deep Learning".
  Several things that I learned in this project are:

1) Working with PyTorch
2) Convolotional Neural Networks
3) Woring with sequential data
4) LSTM Auto-Encoder


## Specific Details
We worked with three differnet datasets.
Synthetic dataset, MNIST and S&P500 stock prices.

For the Synthetic dataset I trained LSTM Auto-Encoder
that took as an input data (sequential data) from the synthetic dataset, 
encode this data to a lower dimention and the decode the 
output of the encoder. 

For the MNIST dataset I trained LSTM Auto-Encoder
that took the images of the MNIST dataset and refer 
to every image as a sequential data (row-by-row).
Then i added the option of classification to the AE
by compose two objective functions. 
Then i repet the same thing but instead of refer 
to every image as row-by-row sequence i did it pixel-by-pixel

For the S&P500 dataset, in addition to train the LSTM AE,
i modified the network to perform a prediction of the 
stock prices.
