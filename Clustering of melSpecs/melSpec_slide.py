import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from os.path import join


def slide_over_image(image_path, widget_width):
    """
    Slides a widget over a very wide image using matplotlib.

    Args:
      image_path: Path to the image file.
      widget_width: Width of the widget in pixels.
    """
    # Load the image
    image = np.transpose(np.load(image_path))

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image, aspect="auto", extent=(0, image.shape[1], 0, image.shape[0]))

    # Create the slider
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, "X Offset", 0, image.shape[1] - widget_width, valinit=0)

    # Update the image view based on slider position
    def update(val):
        # Get the current slider position
        offset = int(slider.val)

        # Create a new view of the image with the desired offset
        view = image[:, offset : offset + widget_width]

        # Update the image display
        ax.imshow(
            view,
            cmap="gray",
            aspect="auto",
            extent=(offset, offset + widget_width, 0, image.shape[0]),
        )
        plt.draw()

    # Connect the slider to the update function
    slider.on_changed(update)

    # Show the plot
    plt.show()


# Example usage
path_input = r"../Dataset_Sentence"
participant = "p07_ses1_sentences"
image_path = join(path_input, f"{participant}_spec.npy")
widget_width = 500  # Adjust based on your widget size

slide_over_image(image_path, widget_width)
