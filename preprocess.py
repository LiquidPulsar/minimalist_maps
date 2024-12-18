import numpy as np
from PIL import Image

yellow = (235, 218, 78)
dark_yellow = (117, 107, 64)
yell_arr = np.array([yellow, dark_yellow])


# def image_to_bool_array(image_path):
#     image = Image.open(image_path)
#     print(image.size)

#     image = image.convert("RGB")
#     np_arr = np.array(image).astype(np.int16)
#     print(np_arr.shape)

#     yellow_range = 30

#     bool_arr = (
#         np.any(
#             np.all(
#                 np.abs(np_arr[:, :, np.newaxis, :] - yell_arr) < yellow_range, axis=3
#             ),
#             axis=2,
#         ).astype(np.uint8)
#         * 255
#     )
#     print("aa")
#     Image.fromarray(bool_arr, mode="L").show()
#     return bool_arr


def image_to_bool_array(image_path):
    image = Image.open(image_path)
    print(image.size)

    image = image.convert("RGB")
    np_arr = np.array(image).astype(np.float32)
    np_arr = np_arr / np_arr.mean(axis=2)[:, :, np.newaxis]

    yellow_range = 0.16

    bool_arr = (
        np.any(
            np.all(
                np.abs(
                    np_arr[:, :, np.newaxis, :]
                    - yell_arr / np.mean(yell_arr, axis=1)[:, np.newaxis]
                )
                < yellow_range,
                axis=3,
            ),
            axis=2,
        ).astype(np.uint8)
        * 255
    )
    # Image.fromarray(bool_arr, mode="L").show()
    return bool_arr


image_path = "maps/image.png"
bool_array = image_to_bool_array(image_path)

reduced_bool_array = bool_array[::10, ::10]
np.save("maps/reduced_bool_array.npy", reduced_bool_array)

# Image.fromarray(reduced_bool_array, mode="L").save("maps/bool_array.png")