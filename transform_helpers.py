import torch
import numpy as np
import cv2


def points_in_circle(x, y, radius, width, height):
    # Initialize an empty list to store the points in the circle
    points = []

    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if (
                (i - x) ** 2 + (j - y) ** 2 <= radius**2
                and i > 0
                and j > 0
                and i < width - 1
                and j < height - 1
            ):
                # Calculate the distance from the center (x, y)
                # If the distance is less than or equal to the radius, both coordinates are non-negative,
                # and the coordinates are within the specified width and height bounds, append the point
                points.append((i, j))

    return points


def merge_scribbles(image, sketch):
    scribbles = get_random_scribble_from_image(image)

    scribbles_mask = scribbles == 0

    return torch.where(scribbles_mask, sketch, scribbles)


def get_random_scribble_from_image(image):
    # Get the image size

    channels, height, width = image.shape

    # Get the radius of the scribble

    num_scribbles = torch.randint(1, 5, (1,))

    scribble_radius = torch.randint(1, 10, (num_scribbles,))

    # get the length of the scribble

    scribble_length = torch.randint(1, 10, (num_scribbles,))

    # get the angle of the scribble

    scribble_angle = torch.randint(0, 360, (num_scribbles,))

    # get the starting position of the scribble

    scribble_start_x = torch.randint(0, width, (num_scribbles,))
    scribble_start_y = torch.randint(0, height, (num_scribbles,))
    points_to_keep = []
    for i in range(num_scribbles):
        for j in range(scribble_length[i]):
            x = int(scribble_start_x[i] + j * torch.cos(scribble_angle[i]))

            y = int(scribble_start_y[i] + j * torch.sin(scribble_angle[i]))

            # check if the point is within the image
            if x < 0 or x >= width or y < 0 or y >= height:
                x = (x + width) % width
                y = (y + height) % height
            points_to_keep.append((x, y))
            points_to_keep.extend(
                points_in_circle(x, y, scribble_radius[i], width, height)
            )
    points_to_keep = list(set(points_to_keep))

    if channels == 3:
        scribble_image = torch.zeros_like(image)
    else:
        rgb = torch.zeros((3, height, width))
        alpha = torch.ones((1, height, width))
        scribble_image = torch.cat((rgb, alpha), dim=0)

    for x, y in points_to_keep:
        scribble_image[:, y, x] = image[:, y, x]

    return scribble_image


def is_in_the_tolerable_level_of_grayscale(image, tolerance=3):
    image = np.array(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False  # Not a 3-channel image

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Compute the standard deviation of the A and B channels
    std_a = np.std(lab_image[:, :, 1])
    std_b = np.std(lab_image[:, :, 2])

    # Calculate the colorfulness metric
    colorfulness = np.sqrt(std_a**2 + std_b**2)

    return colorfulness >= tolerance


def count_white_pixels(img, percentage=0.6):
    # check that the iamge has channels last

    # convert image to numpy array
    img = np.array(img)

    # count the number of white pixels
    white_pixels = np.sum(img == 255)

    # count the number of pixels in the image
    total_pixels = img.shape[0] * img.shape[1]

    # if the percentage of white pixels is lower than the threshold
    # and the image has 3 channels
    return white_pixels / total_pixels <= percentage and img.shape[-1] == 3
