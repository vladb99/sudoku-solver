import numpy as np

import math
import cv2

import imageio
from sudoku import Sudoku

import imgclassifier


def rgb2gray(rgb):
    """Turns an Image with RGB values into an Image with gray values

    Parameters:
    rgb: RGB Image

    Returns:
    gray: Image with gray values
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def invert(img):
    """Inverting the color of the given image

    Parameters:
    img: input image to invert

    Returns:
    inverted_img: Input image but inverted
    """
    height, width = np.shape(img)
    inverted_img = np.zeros((math.ceil(height), math.ceil(width)))
    for x in range(0, height):
        for y in range(0, width):
            inverted_img[x, y] = 255 - img[x, y]
    return inverted_img


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def apply_structure_element(img, filter, iter_num, is_dilation):
    """ Applys structure element to an Image

    Parameters:
    img: Image to which the structure element is applied
    filter: Is the structure element which consists of zeros and ones in our case
    iter_num: How often the structure element is applied
    is_dilation: Flag to distinguish between erode and dilate

    Returns:
    copy: Returns image with applied structure
    
    """
    height, width = np.shape(img)
    N, _ = np.shape(filter)
    K = math.floor(N / 2)
    copy = np.copy(img)
    for _ in range(iter_num):
        padded_img = np.pad(copy, K, pad_with, padder=0)
        mult_list = []
        for x in range(K, padded_img.shape[0] - K):
            for y in range(K, padded_img.shape[1] - K):
                mult_list.append(np.multiply(padded_img[x - K:x + K + 1, y - K:y + K + 1], filter))
        newlist = [np.max(i) if is_dilation else np.min(i) for i in mult_list]
        newlist = np.array(newlist)
        newlist[newlist < 0] = 0
        newlist[newlist > 255] = 255

        return newlist.reshape(height, width)


def erode(img, filter, iter_num):
    """Erode over the input image with a given filter

    Parameters:
    img: Input Image
    filter: Structure Element
    iter_num: How often the structure element is applied

    Returns:
    apply_structure_element(...): Calls method with is_dilation = False
    
    """
    return apply_structure_element(img, filter, iter_num, False)


def dilate(img, filter, iter_num):
    """Dilate over the input image with a given filter

    Parameters:
    img: Input Image
    filter: Structure Element
    iter_num: How often the structure element is applied

    Returns:
    apply_structure_element(...): Calls method with is_dilation = True
    """
    return apply_structure_element(img, filter, iter_num, True)


def region_labeling(img, region_value):
    """Labels every region of the image with a different number beginning from two

    Parameters:
    img: Input image
    region_value: Value that we want to look at

    Returns:
    copy: Image with labeling
    """
    m = 2
    copy = np.copy(img)
    for x, row in enumerate(copy):
        for y, value in enumerate(row):
            if value == region_value:
                flood_fill(copy, x, y, m, region_value)
                m += 1
    return copy


def flood_fill(img, x, y, m, region_value):
    """Fills regions of same value

    Parameters:
    img: Input image
    x: x coordinate of img
    y: y coordinate of img
    m: Value that is given to a pixel based on its region_value
    region_value: Value that we want to look at
    """
    height, width = np.shape(img)
    stack = [(x, y)]
    while len(stack) != 0:
        coord_x, coord_y = stack.pop()
        if 0 <= coord_x < height and 0 <= coord_y < width and img[coord_x, coord_y] == region_value:
            img[coord_x, coord_y] = m
            stack.append((coord_x + 1, coord_y))
            stack.append((coord_x, coord_y + 1))
            stack.append((coord_x, coord_y - 1))
            stack.append((coord_x - 1, coord_y))


def make_img_binary(img, threshold):
    """Convert the image to a binary image

    Parameters:
    img: Input image
    threshold: Value of half the maximum Pixel

    Returns:
    bw_image: Binary image
    
    """
    # converting to its binary form
    _, bw_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return bw_img


def find_biggest_blob(img):
    """Findes the biggest Part which is connected in the image

    Parameters:
    img: Input image

    Returns:
    biggest_blob: Image where only the biggest blob is visible
   """
    biggest_blob = np.copy(img)
    unique, counts = np.unique(biggest_blob, return_counts=True)
    dict = np.asarray((unique, counts)).T
    # Get rid of label 0, which is the background
    dict = dict[1:]
    # Sort the label by occurences
    most_frequent_label = sorted(dict, key=lambda t: t[1])[-1][0]
    biggest_blob[biggest_blob != most_frequent_label] = 0
    return biggest_blob


def get_mask(img):
    """Get the mask of the Sudoku field from the image

    Parameters:
    img: Input image

    Returns:
    mask: Mask of our Sudoku field in white the rest in black
    
    """
    mask = region_labeling(img, 0)
    unique, _ = np.unique(mask, return_counts=True)
    first_label = unique[0]
    mask[mask != first_label] = 255
    return mask


def sobel_operator():
    """Builds the Sobel operators for x and y

    Returns:
    Hx: Sobel operator for x
    Hy: Sobel operator for y
    """
    Hx = np.array([[1], [2], [1]]) * np.array([[-1, 0, 1]])
    Hy = np.array([[1, 2, 1]]) * np.array([[-1], [0], [1]])
    return Hx, Hy


def apply_sobel_filter(img, sobel_filter, mask):
    """Applying the sobel operators to the input image

    Parameters:
    img: Input image
    sobel_filter: Hx and Hy
    mask: Image where the Sudoku field is white and the rest black

    Returns:
    copy: Image with sobel operators applied
    """

    height, width = np.shape(img)

    # NxN Filter
    N, _ = np.shape(sobel_filter)

    # 2K + 1 = N
    K = math.floor(N / 2)

    padded_img = np.pad(img, K, pad_with, padder=0)
    mult_list = []
    for x in range(K, padded_img.shape[0] - K):
        for y in range(K, padded_img.shape[1] - K):
            if mask[x - K, y - K] == 0:
                mult_list.append(0)
                continue
            mult_list.append(np.sum(np.multiply(padded_img[x - K:x + K + 1, y - K:y + K + 1], sobel_filter)))
    mult_list = np.array(mult_list)
    mult_list = np.around(mult_list, 0)
    mult_list[mult_list < 0] = 0
    mult_list[mult_list > 255] = 255
    return mult_list.reshape(height, width)


def closing(img, filter):
    """First dilate and then erode

    Parameters:
    img: Input image
    filter: Structure element

    Returns:
    erode: dilated and eroded image
    """
    img = dilate(img, filter, 1)
    return erode(img, filter, 1)


def opening(img, filter):
    img = erode(img, filter, 1)
    return dilate(img, filter, 1)


def get_sudoku_lines(img):
    """Get only white field lines from given image. Ignore digits by drawing them black.

    Parameters:
    img: Input image

    Returns:
    lines_img: Image containing only lines and not digits.
    """
    lines_img = region_labeling(img, 1)
    unique, counts = np.unique(lines_img, return_counts=True)
    dict = np.asarray((unique, counts)).T
    # Get rid of label 0, which is the background
    dict = dict[1:]
    # Sort the label by occurences and only get last 10 elements
    sorted_labels = sorted(dict, key=lambda t: t[1])
    most_frequent_labels = sorted_labels[-10:]
    rest_labels = sorted_labels[:-10]
    # Make the most frequent labels white and the rest black

    for array in most_frequent_labels:
        lines_img[lines_img == array[0]] = 255

    for array in rest_labels:
        lines_img[lines_img == array[0]] = 0
    return lines_img


def reduce_regions(img):
    """Reduces regions of white pixel to only one point.

    Parameters:
    img: Input image

    Returns: Tuple of regions_img and points
    """
    regions_img = region_labeling(make_img_binary(img, 1), 1)
    unique, _ = np.unique(regions_img, return_counts=True)
    # Remove the background
    unique = np.delete(unique, 0)
    points = []
    for label in unique:
        # Get all points with label
        label_region = np.transpose(np.where(regions_img == label))
        # Set all these label points to black
        regions_img[regions_img == label] = 0
        # Get the centroid of the region
        mean = label_region.mean(0)
        points.append(mean)
        # Set the centroid to white
        regions_img[int(mean[0]), int(mean[1])] = 1
    return (regions_img, points)


def sort_grid_points(points):
    """Sorts points from upper left corner to bottom right corner.

    Parameters:
    points: List of grid points to be sorted.

    Returns sorted list.
    """

    c = np.array(points).reshape((100, 2))
    c2 = c[np.argsort(c[:, 1])]

    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    bm = b.reshape((10, 10, 2))
    return bm


def prepare_points_pairs_for_transformation(columns):
    """Divides series of intersection points into pairs of four points.

    Parameters:
    columns: List of columns. Each column contains 10 vertical points from the image.

    Returns list of four points pairs.
    """

    four_points_pairs = []
    for x, column in enumerate(columns):
        # last column reached:
        if x == 9:
            continue
        for y, point in enumerate(column):
            # last value in column reached
            if y == 9:
                continue

            # Get the points from own and next column, to form the rectangle
            point2 = columns[x + 1][y]
            point3 = columns[x][y + 1]
            point4 = columns[x + 1][y + 1]
            four_points_pairs.append([point, point2, point3, point4])
    return np.array(four_points_pairs, dtype=np.float32)


def preprocess_img(img):
    """Preprocesses image with adaptive threshold and inverting.

    Parameters:
    img: Gray image as numpy array.

    Returns:
    inverted_img, adapted_threshold_img: Returns a tuple with inverted image and adaptive threshholded image.
    """

    element = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    morphologic_closing_img = closing(img, element)
    # Kinda ugly but needs to be done, because we can't divide by 0
    # and the dilation algorithm chooses 0 for pixel over the edge
    morphologic_closing_img[morphologic_closing_img == 0] = 255
    adapted_threshold_img = img / morphologic_closing_img * 0.85
    adapted_threshold_img_binary = make_img_binary(adapted_threshold_img, 0.7)
    imageio.imwrite("sudoku-project/pipeline-images/step0.png", adapted_threshold_img)
    imageio.imwrite("sudoku-project/pipeline-images/step1.png", adapted_threshold_img_binary)
    print("Step 0 and 1: Adaptive Threshold done")

    # Inverting
    inverted_img = invert(adapted_threshold_img_binary)
    imageio.imwrite("sudoku-project/pipeline-images/step2.png", inverted_img)
    print("Step 2: Invert done")

    return inverted_img, adapted_threshold_img


def get_sudoku_mask(img):
    """Calculates the mask for the sudoku board, by doing region labeling and then finding the biggest blob.

    Parameters:
    img: Inverted binary image as numpy array.

    Returns:
    mask: Returns an image where all pixels inside the sudoku board are white.
    """
    # Region labeling with flood filling
    # IF ANYTHING GOES WRONG CHECK IF THE 254 is the minimum in inverted_img
    flood_filled_img = region_labeling(make_img_binary(img, 254), 1)
    imageio.imwrite("sudoku-project/pipeline-images/step3.png", flood_filled_img)
    print("Step 3: Region labeled done")

    # Finding biggest blob based on labels
    sudoku_board_img = find_biggest_blob(flood_filled_img)
    imageio.imwrite("sudoku-project/pipeline-images/step4.png", sudoku_board_img)
    print("Step 4: Biggest blob done")

    # Getting sudoku board mask with flood filling
    mask = make_img_binary(get_mask(sudoku_board_img), 128)
    imageio.imwrite("sudoku-project/pipeline-images/step5.png", mask)
    print("Step 5: Mask done")

    return mask


def detect_lines(adapted_threshold_img, mask):
    """Detects the lines by applying sobel filters. Disjoint lines are fixed by dilating vertically and horizontally.

    Parameters:
    adapted_threshold_img: Threshhold adapted image as numpy array.
    mask: Mask area where sobel filters to be applied.

    Returns:
    vertical_lines_img, horizontal_lines_img: Returns images containing only vertical or horizontal lines.
    """

    # Apply sobel filters
    s_vert, s_hor = sobel_operator()
    sobel_vertical_img = apply_sobel_filter(adapted_threshold_img, s_vert, mask)
    sobel_horizontal_img = apply_sobel_filter(adapted_threshold_img, s_hor, mask)
    sobel_vertical_img = make_img_binary(sobel_vertical_img, 0)
    sobel_horizontal_img = make_img_binary(sobel_horizontal_img, 0)
    imageio.imwrite("sudoku-project/pipeline-images/step6.png", sobel_vertical_img)
    imageio.imwrite("sudoku-project/pipeline-images/step7.png", sobel_horizontal_img)
    print("Step 6 and 7: Vertical and horizontal line detection done")

    # Dilate the found vertical and horizontal lines, so they aren't disjoint anymore
    structure_element_horizontal = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    dilated_horizontal_img = dilate(sobel_horizontal_img, structure_element_horizontal, 1)

    structure_element_vertical = structure_element_horizontal.transpose()
    dilated_vertical_img = dilate(sobel_vertical_img, structure_element_vertical, 1)

    imageio.imwrite("sudoku-project/pipeline-images/step8.png", dilated_vertical_img)
    imageio.imwrite("sudoku-project/pipeline-images/step9.png", dilated_horizontal_img)
    print("Step 8 and 9: Fixing disjoint lines done")

    # Using regions to ignore the digits
    vertical_lines_img = get_sudoku_lines(dilated_vertical_img)
    horizontal_lines_img = get_sudoku_lines(dilated_horizontal_img)

    vertical_lines_img = dilate(vertical_lines_img, structure_element_vertical, 1)
    horizontal_lines_img = dilate(horizontal_lines_img, structure_element_horizontal, 1)
    imageio.imwrite("sudoku-project/pipeline-images/step10.png", vertical_lines_img)
    imageio.imwrite("sudoku-project/pipeline-images/step11.png", horizontal_lines_img)
    print("Step 10 and 11: Getting lines without digits finished")

    return vertical_lines_img, horizontal_lines_img


def get_points_for_transformation(vertical_img, horizontal_img):
    """Calculates the intersection points out of the vertical and horizontal images.

    Parameters:
    vertical_img: Image as numpy array containing only vertical lines.
    horizontal_img: Image as numpy array containing only horizontal lines.

    Returns:
    src, dst: Returns src and dst, where each contain pairs of four points. src are the intersection points, which are transformed to dst.
    """

    # Get the intersect points of the lines
    intersected_regions_img = cv2.bitwise_and(vertical_img, horizontal_img)
    imageio.imwrite("sudoku-project/pipeline-images/step12.png", intersected_regions_img)
    print("Step 12: Getting intersection regions done")

    # Reduce the intersect regions down to only one point
    centroids_img, points = reduce_regions(intersected_regions_img)
    imageio.imwrite("sudoku-project/pipeline-images/step13.png", centroids_img)
    print("Step 13: Getting intersection points done")

    # Sort points on the grid
    sorted_points = sort_grid_points(points)

    # Get all points between 0 and 450
    interpolation = np.linspace(0, 450, num=10, endpoint=True)
    columns = []
    for x in interpolation:
        columns.append([list(a) for a in zip(np.full(10, x), interpolation)])
    columns = np.array(columns)

    # flip x and y bacause of opencv
    for x, row in enumerate(sorted_points):
        for y, _ in enumerate(row):
            sorted_points[x, y] = np.flip(sorted_points[x, y])
            columns[x, y] = np.flip(columns[x, y])

    # change the order in the columns array from rows to columns
    columns_modified = []
    for i in range(10):
        columns_modified.append(columns[:, i])
    columns_modified = np.array(columns_modified)

    # divide into pairs of four points
    src = prepare_points_pairs_for_transformation(sorted_points)
    dst = prepare_points_pairs_for_transformation(columns_modified)
    print("Preparing points done")

    return (src, dst)


def transform_intersection_points(img, src, dst):
    """Calculates warp perspective for all sudoku cells.

    Parameters:
    img: Binary image as numpy array.
    src: Pairs of points from intersection points.
    dst: Pairs of points from interpolation points.

    Returns:
    cells: Returns a list of inverted images of each sudoku cell.
    """

    board_croped_img = np.zeros((450, 450))
    cells = []
    for src_point, dst_point in zip(src, dst):
        matrix = cv2.getPerspectiveTransform(src_point, dst_point)
        warp = cv2.warpPerspective(img, matrix, (450, 450))

        cell = np.zeros((50, 50))
        i = 0
        # Warp gets an image of the sudoku board, where each sell is in the right perspective.
        # We take that perspective of each cell, and place it in our output
        for x in range(int(dst_point[0][0]), int(dst_point[3][0])):
            j = 0
            for y in range(int(dst_point[0][1]), int(dst_point[3][1])):
                board_croped_img[y, x] = warp[y, x]
                cell[j, i] = warp[y, x]
                j += 1
            i += 1

        _, thresh = cv2.threshold(cell, 0.5, 1, cv2.THRESH_BINARY)
        cell = invert(thresh)
        cells.append(make_img_binary(cell, 254))

        # Don't delete this because it shows that the warpPerspective works
        # Watch down the the cells columnwise not rowwise!
        # plt.subplot(211)
        # plt.imshow(cell, cmap=cm.Greys_r)
        # plt.show()
    imageio.imwrite("sudoku-project/pipeline-images/step14.png", board_croped_img)
    print("Step 14: Transforming points done")
    return cells


def remove_edge(cell):
    cell = np.pad(cell[5:-5, 5:-5], pad_width=((1, 1), (1, 1)), mode='constant',
                  constant_values=0)
    return cell


if __name__ == "__main__":
    # read img
    img = cv2.imread("sudoku-project/board-images/vorfuehrung1.jpeg")
    #img = cv2.imread("board-images/vorfuehrung2.jpeg")
    #img = cv2.imread("board-images/vorfuehrung3.jpeg")

    if img.shape[0] > 450:
        width = 450
        height = 450
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("sudoku-project/pipeline-images/test.png", img)

    img = np.array(img)
    img_gray = rgb2gray(img)

    inverted_img, adaptive_threshold_img = preprocess_img(img_gray)
    mask_img = get_sudoku_mask(inverted_img)
    vertical_lines_img, horizontal_lines_img = detect_lines(adaptive_threshold_img, mask_img)
    intersection_points, destination_points = get_points_for_transformation(vertical_lines_img, horizontal_lines_img)
    sudoku_cells = transform_intersection_points(adaptive_threshold_img, intersection_points, destination_points)
    element = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # sudoku_cells = [dilate(cell, element, 1) for cell in sudoku_cells]
    sudoku_cells = [remove_edge(x) for x in sudoku_cells]

    result = imgclassifier.classify_cells(sudoku_cells)

    print(result)
    puzzle = Sudoku(3, 3, board=result.tolist())
    print(puzzle)
    puzzle.solve().show_full()
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(img, cmap=cm.Greys_r)
    # plt.figure(1)
    # plt.subplot(212)
    # plt.imshow(preprocessed_img, cmap=cm.Greys_r)
    # plt.show()
