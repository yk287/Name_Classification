
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#TODO Confusion Matrix

# Set up plot
def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix

    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()





