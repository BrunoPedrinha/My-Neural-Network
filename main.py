from ImgLoader import ImgLoader
from neuraltest import NeuralNetwork
import numpy

tinputs = numpy.array([[0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
toutputs = numpy.array([[0, 1, 1, 0]]).T
rand_arr = numpy.array([ [0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1],
    [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
        [1, 1, 1, 1]])
ans = numpy.array([[0,0,1,1,0,0,0,0,1,1,1,1,1]]).T
run_program = True

nn = NeuralNetwork(tinputs, toutputs)
img = ImgLoader()

if __name__ == "__main__":
    
    test_arr = numpy.random.random((tinputs.shape)) - 1 
    print(test_arr)
    nn.training_network()
    #rand_num = numpy.random.randint(0, len(rand_arr)) 

    while(run_program):
        
        rand_num = numpy.random.randint(0, len(rand_arr))
        rand_inum = numpy.random.randint(0, 4)
        print("random array = " + str(rand_arr[rand_num]))
        new_test = nn.sig_output(rand_arr[rand_num])
        print("Expected answer: " + str(ans[rand_num]))
        print("Bot answer: " + str(new_test))
        
        if new_test <= 0.5:
            img.Load_Image('C:/Users/Bruno/Desktop/MyNeuralNetwork/Bot Imgs/dog'+str(rand_inum)+'.jpg')    
        else:
            img.Load_Image('C:/Users/Bruno/Desktop/MyNeuralNetwork/Bot Imgs/cat'+str(rand_inum)+'.jpg')
        in_ans = input("Do you want to run it again? (y/n) ")
        if in_ans == "y" or in_ans.lower().startswith("y"):
            pass
        else:
            run_program = False


#test = numpy.random.rand(4,1)
#print(test)
