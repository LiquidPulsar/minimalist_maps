import numpy as np
from numpy._typing._array_like import NDArray
from sklearn.metrics import f1_score
from PIL import Image

Genome = list[tuple[int, int, int]]  # [(x, y, r), ...]


def borders_of_continents(arr):
    return (arr[1:, :-1] != arr[:-1, :-1]) | (arr[1:, 1:] != arr[1:, :-1])


def display(arr):
    Image.fromarray(arr.astype(np.uint8) * 255, mode="L").show()


def display_green_blue(arr, lines=None):
    colored_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    colored_arr[arr] = [0, 255, 0]  # Green for True
    colored_arr[~arr] = [0, 0, 255]  # Blue for False
    if lines is not None:
        colored_arr[lines] = [0, 0, 0]
    Image.fromarray(colored_arr).show()


class Phenotype:
    def __init__(self, genome: Genome, image_dim: tuple[int, int]):
        self.h, self.w = image_dim
        self.genome = genome
        self.masks = np.array(
            [self.create_circular_mask((x, y), r) for x, y, r in genome]
        )

    def create_circular_mask(self, center: tuple[int, int], radius: int) -> np.ndarray:
        Y, X = np.ogrid[: self.h, : self.w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        # Image.fromarray(mask.astype(np.uint8) * 255, mode="L").show()
        return mask

    def classify(self, image: np.ndarray):
        # O(n * m * 2^m) where n is the number of pixels and m is the number of circles
        arr = np.zeros((self.h, self.w), dtype=bool)
        for i in range(2 ** self.masks.shape[0]):
            region = self.get_region(i)
            if region.sum() > 0 and image[region].mean() > 0.5:
                arr |= region
        return arr

    def get_region(self, i: int):
        arr = np.ones((self.h, self.w), dtype=bool)
        for j in range(self.masks.shape[0]):
            if i & (1 << j):
                arr &= self.masks[j]
            else:
                arr &= ~self.masks[j]
        return arr

    def score(self, image: np.ndarray) -> float:
        # np.arrya to convert from jax array
        return f1_score(image.ravel(), self.classify(image).ravel())  # type: ignore

    def circle_borders(self):
        return np.array(
            [borders_of_continents(mask) for mask in self.masks], dtype=bool
        )

    def __hash__(self) -> int:
        return hash(tuple(self.genome))
