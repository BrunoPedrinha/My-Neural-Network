import matplotlib.pyplot as plt
import matplotlib.image as mimg
import os

class ImgLoader:
    
    def __init__(self):
        pass

    def Check_Path(self, my_path):
        if not os.path.isdir(my_path):
            print("NOT A PATH")
            return False
        elif len(os.listdir(my_path)) == 0:
            print("NOTHING IN THE FOLDER/PATH")
            return False
        elif not any(imgs.endswith('.jpg') for imgs in os.listdir(my_path)):
            print(imgs)
            return False
        return True
    
    def Find_Path(self, path):
        txt = path.rfind("/")
        new_path = path[:txt]
        return new_path

    def Load_Image(self, img_path):
        new_path = self.Find_Path(img_path)
        if self.Check_Path(new_path):
            plt.ion()
            img = mimg.imread(img_path)
            imgp = plt.imshow(img)
            plt.show()
            plt.pause(5)
            plt.close()

    def Close_Image(self):
        plt.pause(3)
        plt.close()
