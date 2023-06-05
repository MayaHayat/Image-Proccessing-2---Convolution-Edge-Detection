import math

import numpy as np
import cv2

def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 12345

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    reversedK_size = np.flip(k_size)
    padded_array = np.pad(in_signal, (k_size.size-1, k_size.size-1), mode='constant')
    #print(padded_array)
    result = [0]*(in_signal.size + k_size.size -1)
    #print(result)
    for i in range (in_signal.size + k_size.size -1):
        sum = 0
        for j in range (k_size.size):
            sum+= (padded_array[i+j]*reversedK_size[j])
        result[i] = sum
    #print(result)

    return result



def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """


    height, width = kernel.shape
    padded_mat = np.pad(in_image, ((height, height), (width, width)), 'constant')

    # # Flip on x axis
    xfliped = np.flipud(kernel)

    # Flip on y axis
    flipped = np.fliplr(kernel)
    out_image = np.zeros_like(in_image)

    # Iterate over the input image
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            # Calculate the convolution
            out_image[i, j] = np.sum(padded_mat[i:i + height, j:j + width] * kernel)

    return out_image

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    # As we saw in class using (-1 0 1) as the kernel would give us the vertical edges and (-1 0 1)^T gives us the horizonal edges
    kernel_horizontal = np.array([[1, 0, -1]])
    kernel_vertical = np.array([[1, 0, -1]]).transpose()
    #print(kernel_horizontal)
    #print(kernel_vertical)
    d_x = conv2D(in_image, kernel_horizontal)
    d_y = conv2D(in_image, kernel_vertical)

    magnitude = np.sqrt(np.power(d_x, 2) + np.power(d_y, 2))
    direction = np.arctan2(d_y, d_x)

    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    sigma = 1
    centre = k_size // 2
    kernel_mat = np.zeros((k_size, k_size))
    for i in range (k_size):
        for j in range (k_size):
            x = i - centre
            y = j - centre
            kernel_mat[i,j] = np.exp(-(np.power(x,2) + np.power(y,2))/(2 * np.power(centre, 2)))
    gaussian_kernel = kernel_mat / sigma
    img_blurry = conv2D(in_image, gaussian_kernel)
    return img_blurry


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    sigma = 1
    gaussian = cv2.getGaussianKernel(k_size, sigma)
    img_blurry = cv2.sepFilter2D(in_image, -1, gaussian, gaussian)

    return img_blurry


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # chose 9 as it shows it the best
    img = blurImage1(img, 7)
    img_log = cv2.Laplacian(img, cv2.CV_64F)

    # Initialize the edge matrix with zeros
    edge_matrix = np.zeros_like(img_log)
    # Loop over the image pixels
    for i in range(1, img_log.shape[0] - 1):
        for j in range(1, img_log.shape[1] - 1):
            # Get the neighboring pixels
            neighbors = img_log[i - 1 : i + 1, j - 1:j + 1]
            # Check if there is a sign change
            if np.max(neighbors) > 0 and np.min(neighbors) < 0:
                edge_matrix[i, j] = 1
    return edge_matrix





def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Open CV function: cv.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    img = (img * 255).astype(np.uint8)
    edge = cv2.Canny(img, 75, 200)

    r_range = len(range(min_radius, max_radius))
    # Initialize the accumulator array
    H = np.zeros((img.shape[0], img.shape[1], r_range + 1))
    #print(len(np.where(edge)))
    # Iterates over the height and length to check each pixel
    for y in range(edge.shape[0]):
        for x in range(edge.shape[1]):
            # Checks if it is an edge
            if edge[y, x] > 0:
                for currRad in range(min_radius, max_radius + 1):
                    #let the centre of the circle be (x_c,y_c)
                    x_c = x + currRad * np.cos(np.deg2rad(np.arange(0, 360)))
                    y_c = y + currRad * np.sin(np.deg2rad(np.arange(0, 360)))
                    valid_points = (x_c >= 0) & (x_c < edge.shape[1]) & (y_c >= 0) & (y_c < edge.shape[0])
                    x_valid = x_c[valid_points]
                    y_valid = y_c[valid_points]
                    np.add.at(H, (y_valid.astype(int), x_valid.astype(int), currRad - min_radius), 1)

    # Find local maxima in the accumulator
    circles = []
    for currRad in range(max_radius - min_radius + 1):
        print(currRad)
        for y in range(edge.shape[0]):
            for x in range(edge.shape[1]):
                if H[y, x, currRad] > (np.max(H)/1.5):
                    circles.append((x, y, currRad + min_radius))
    return circles




def bilateral_filter_implement(in_image, k_size, sigma_s, sigma_r):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: Open CV implementation, my implementation
    """

    cv2_answer = cv2.bilateralFilter(in_image.astype(np.uint8), k_size, sigma_s, sigma_r)

    img32 = in_image.astype(np.float32)
    my_answer = np.zeros_like(img32)
    # As we begin by denoising
    padded_image = np.pad(in_image, k_size // 2, mode='constant')

    spatial_kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            spatial_kernel[i, j] = np.exp(-((i - k_size // 2) ** 2 + (j - k_size // 2) ** 2) / (2 * sigma_s ** 2))

    # Compute the range kernel and apply the bilateral filter
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            midPix = padded_image[i:i+k_size, j:j+k_size]
            intensity_kernel = np.exp(-((midPix - in_image[i, j]) ** 2) / (2 * sigma_r ** 2))
            weights = spatial_kernel * intensity_kernel
            weightsNorm = weights / np.sum(weights)
            my_answer[i, j] = np.sum(weightsNorm * midPix)

    # Convert the filtered image back to uint8 format
    my_filtered = my_answer.astype(np.uint8)

    return cv2_answer, my_filtered