This is my first Neural Network and introduction to AI! I'm using the numpy library in python to create matrices and compute math with the matrices. This neural network uses the Sigmoid activation function. I chose the Sigmoid activation because I only need an output between 0 and 1 since I'm only using binary matrices and I'm not doing anything overly complicated yet. I feed the neural network 4 1x4 matrices. If the matrix has a 1 in the first column ([1,0,0,0]) then the bots answer should be as close to 1 as possible, but if the matrix has a 0 in the first column ([0,1,0,1]) then the bots answer should be as close to 0 as possible. The neural network trains off these 4 examples 60,000 times using matrix multiplication coupled with forward propagation and backpropagation to fix the error margin so the bot's answer is closer to the right answer, and showing the error margin after intervals of 10,000 so you can see how effective the training has been. Once training is done i have 12 4x1 matrices and the bot picks 1 at random, gives an expected answer, and then gives the bot answer. If the answer is greater than 0.5 a cat picture will be shown, otherwise a dog picture will be shown. 

In the above video, the error margin should be close to [0, 1, 1, 0], as you can see the bot is learning and fixing the error margin every training session.