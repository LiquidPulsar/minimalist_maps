import glob, os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directory containing the PNG files
directory = r"C:\Users\Alex\OneDrive\Desktop\Random\Python\maps\results"

# Create a list of file paths for the PNG files
file_paths = sorted(glob.glob(os.path.join(directory,"*.png")),key=lambda x: int(x.rsplit("_",1)[1].split(".")[0]))

print(file_paths)

# Create a figure and axis
fig, ax = plt.subplots()


# Function to update the frame
def update(frame):
    img = plt.imread(file_paths[frame])
    ax.imshow(img)
    ax.axis("off")


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(file_paths), repeat=False) # type: ignore

# Save the animation as a GIF
ani.save("maps/genetic_algorithm_animation.gif", writer="imagemagick")

plt.show()
