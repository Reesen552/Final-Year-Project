
from matplotlib import pyplot as plt
import numpy as np

def dispImg(img,title = 'picture'):
    '''Because of your images which were loaded by opencv, 
    in order to display the correct output with matplotlib, 
    you need to reduce the range of your floating point image from [0,255] to [0,1] 
    and converting the image from BGR to RGB:'''

    img = np.array(img,dtype=float)/float(255)
    img = img[:,:,::-1]

    plt.imshow(img)
    plt.title(title)
    plt.show()
    return 0