import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.figure as fig

# This function is for adding parameters on generated images
def add_label(idx,real_parameter,predict_parameter):
    image = mpimg.imread(str(idx)+'.png')
    fig = plt.figure( 1 )
    ax = fig.add_subplot( 111 )
    ax.set_title("GTA V Predicting")

    im = ax.imshow(np.zeros((128, 256*2, 3)))
    
    im.set_data(image)
    im.axes.figure.canvas.draw()
    predict = "Steering:%f\nThrottle:%f" % (predict_parameter[3],predict_parameter[4])
    real = "Steering:%f\nThrottle:%f" % (real_parameter[3],real_parameter[4])
    txt = ax.text(20,30,real,style='italic',fontsize=7,
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    txt2 = ax.text(270,30,predict,style='italic',fontsize=7,
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    plt.savefig(str(idx)+'_label.png')

def main():# for testing
	target_parameter = np.ones(6)
	predict_paramter = target_parameter
	display(1,target_parameter,predict_paramter)
	
if __name__ == "__main__":
	main()
