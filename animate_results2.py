import glob, os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directory containing the PNG files
directory = r"C:\Users\Alex\OneDrive\Desktop\Random\Python\maps\results"

with open("maps/results/best_genome.txt") as f:
    genomes = f.readlines()

frames = []
curr = ""
for i, genome in enumerate(genomes):
    if genome != curr:
        curr = genome
        frames.append((i,os.path.join(directory,f"genetic_algorithm_{i}.png")))


# Create a figure and axis
fig, ax = plt.subplots()


# Function to update the frame
def update(frame):
    i, file = frames[frame]
    img = plt.imread(file)
    ax.set_title(f"Generation {i}")
    ax.imshow(img)
    ax.axis("off")


# Set frames per second
fps = 10

# Create the animation with the specified fps
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, repeat=False) # type: ignore
# Save the animation as a GIF
# ani.save("maps/genetic_algorithm_animation.gif", writer="imagemagick")

plt.show()
