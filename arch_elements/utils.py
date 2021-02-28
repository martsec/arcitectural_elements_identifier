import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_samples(x, y, encoder, images_to_print=10, predict_model=None,):
    import random
    axis_width = 5
    axis_height = images_to_print // axis_width

    _, axs = plt.subplots(axis_height, axis_width, figsize=(axis_width*4,axis_height*4))
    for ay, ax in np.ndindex((axis_height,axis_width)):

        sample = random.randint(0, len(x))
        image = x[sample]
        axs[ay, ax].imshow(image.astype('uint8'))
        axs[ay, ax].set_frame_on(False)
        axs[ay, ax].axes.get_yaxis().set_visible(False)
        axs[ay, ax].axes.get_xaxis().set_visible(False)

        true_label = encoder.inverse_transform(y[sample])
        title = "True label: " + true_label
        title_color = 'black'

        if predict_model:
            prediction_scores = predict_model.predict(np.expand_dims(image, axis=0))
            predicted_index = np.argmax(prediction_scores)
            predicted_label = encoder.get_class_string_from_index(predicted_index)
            title += "\nPredicted label: " + predicted_label
            if predicted_label == true_label:
                title_color = 'green'
            else: 
                title_color = 'red'

        axs[ay, ax].set_title(title, color=title_color)

def plot_confusion_matrix(true_labels, predicted_labels, category_names:list, fig_file_name=None, normalize=None):
    cm = confusion_matrix(true_labels, predicted_labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    _, ax = plt.subplots(figsize=(15,15))
    disp.plot(ax=ax, cmap='GnBu')
    if fig_file_name:
        plt.savefig(fig_file_name)