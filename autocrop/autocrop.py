#!/usr/bin/env python3
import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import cv2 as cv
import largestinteriorrectangle as lir
import numpy as np
from timeout_decorator import timeout


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """

    def add_filename(self, filename):
        self._filename = filename

    def filter(self, record):
        record.filename = self._filename
        return True


def find_contours(img):
    """
    Find contours on the image.

    Args:
        img (np.ndarray)    : A numpy array of the image.

    Returns:
        list                : A list of contours found in the image.
    """
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def dilation_and_findcontours(img, iterations, get_img=False):
    """
    Perform dilation and findcontours. Use to find the 96-well plates.

    Args:
        img (np.ndarray)    : A numpy array of the image.
        iterations (int)    : The iteration of dialation. (default=10)
        get_img (bool)      : Return the 2D numpy array of the final image. (default=False)
    Returns:
        list                : A sorted list of contours found in the image.
        np.ndarray|None     : Return a 2D numpy array if get_img=True. Otherwise return None.
    """
    kernel = np.ones((9, 9), np.uint8)
    dilation = cv.dilate(img, kernel, iterations=iterations)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations=10)

    contours = find_contours(closing)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

    if not get_img:
        closing = None
    return sorted_contours, closing


def scale_image(img, adj_width=4000, distortion_offset=1, get_scal_f=False):
    """
    Scale the image by a given width. Specify a float to the distortion_offset to fix the distortion.

    Args:
        img (np.ndarray)            : A numpy array of the image.
        adj_width (int)             : The target width of after adjustment. (default=4000)
        distortion_offset (float)   : Use to fix the image distortion. Set the value to 1 to disable the fix; < 1 to shorten the
                                      height; > 1 to increase the height. (default=1)
        get_scal_f (bool)           : Return the scaling factor.
    Returns:
        np.ndarray                  : A 2D numpy array of the scaled image.
        (float)                     : Return scaling factor if get_scal_f=True.
    """
    scaling_factor = adj_width / img.shape[1]
    adj_height = int(img.shape[0] * scaling_factor * distortion_offset)
    scaled_dim = (adj_width, adj_height)
    img = cv.resize(img, scaled_dim, interpolation=cv.INTER_AREA)
    logging.debug(f"Image shape after scaled: {img.shape}")

    if get_scal_f:
        return img, scaling_factor
    else:
        return img


def fill_container(gray_img):
    """
    Fill the container with white(255) on a grayscaled image to better delineate the contour of the container.

    Args:
        gray_img (np.ndarray)   : A 2D numpy array of the grayscaled image.

    Returns:
        np.ndarray              : A 2D numpy array of the threshold image.
    """
    blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    adapt_th = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    th1 = cv.morphologyEx(adapt_th, cv.MORPH_OPEN, kernel, iterations=1)
    _, th2 = cv.threshold(blur, 50, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    overlap = cv.bitwise_or(th1, th2)
    contours = find_contours(overlap)
    fill_image = cv.drawContours(overlap, contours, -1, color=255, thickness=cv.FILLED)
    return fill_image


@timeout(120)
def detect_largest_inner_rectangle(img):
    """
    Find the largest inner rectangle. Use to get the container. Return the x, y, width, height of the container.

    Args:
        img (np.ndarray)    : A numpy array of the image.

    Returns:
        int                 : x. The x axis start point of the region.
        int                 : y. The y axis start point of the region.
        int                 : w. The width of the region.
        int                 : h. The height of the region.
    """
    contours = find_contours(img)
    largest_contour = max(contours, key=cv.contourArea)

    epsilon = 0.01 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)
    d1, d2, d3 = approx.shape

    x, y, w, h = lir.lir(approx.reshape((d2, d1, d3)))
    return x, y, w, h


def get_markers(hsv_img, lower_bound, upper_bound):
    """
    Find the markers with a given hsv upper and lower bound. Return the x, y, width, height of the markers.

    Args:
        hsv_img (np.ndarray)        : A 3D numpy array of the hsv image.
        lower_bound (np.ndarray)    : A 1D numpy array of the lower bound of each of the color channel R, G and B.
        upper_bound (np.ndarray)    : A 1D numpy array of the upper bound of each of the color channel R, G and B.

    Returns:
        int                         : x. The x axis start point of the region.
        int                         : y. The y axis start point of the region.
        int                         : w. The width of the region.
        int                         : h. The height of the region.
    """
    mask = cv.inRange(hsv_img, lower_bound, upper_bound)
    _, th = cv.threshold(mask, 90, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=1)
    dilation = cv.dilate(opening, kernel, iterations=2)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations=5)

    logging.debug("Detecting markers...")
    contours = find_contours(closing)
    max_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
    x, y, w, h = cv.boundingRect(max_contour)
    logging.debug("Got image that contains markers only.")
    return x, y, w, h


def shrink_image(img, perc=4):
    """
    Shrink the image to remove the edge of the container.

    Args:
        rgb_img (np.ndarray)    : A numpy array of the image.

    Returns:
        np.ndarray              : A numpy array of the shrunken image.
    """
    perc = perc / 100 / 2  # Divide by 2 since we will shrink on both ends of each dimension
    h, w, _ = img.shape
    h = int(h * perc)
    w = int(w * perc)
    img = img[h:-h, w:-w]
    return img


def increase_contrast(rgb_img):
    """
    Increase the contrast to enhance the signal of the colonies.

    Args:
        rgb_img (np.ndarray)    : A 3D numpy array of the rgb image.

    Returns:
        np.ndarray              : A 3D numpy array of the enhanced rgb image.
    """
    lab = cv.cvtColor(rgb_img, cv.COLOR_RGB2LAB)
    l_channel, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv.merge((cl, a, b))
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2RGB)
    return enhanced_img


def find_proper_threshold(gray_img, signal_ratio_target, signal_ratio_diff, threshold=100, epochs=10):
    """
    Find a proper thershold to highlight the colonies. Failed to find a proper param will terminate the program. Failed to find
    a proper param will terminate the program.

    Args:
        gray_img (np.ndarray)       : A 2D numpy array of the grayscaled image.
        signal_ratio_target (float) : The target of signal ratio (signaling pixels / all pixels) of the plate.
        signal_ratio_diff (float)   : The allowing range of the signal_ratio_target.
        threshold (int)             : Initial setting of the threshold. (default=10)
        epochs (int)                : The maximum round of searching. (default=10)

    Returns:
        np.ndarray                  : A 2D numpy array of the threshold image.
    """
    logging.debug(
        "Finding a threshold that have signal ratio between "
        f"{signal_ratio_target - signal_ratio_diff:.2f}-{signal_ratio_target + signal_ratio_diff:.2f} ..."
    )
    signal_ratio = 0
    trend = deque(maxlen=2)
    rate = 1
    for epoch in range(epochs):
        _, th = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)
        signal_ratio = (th > 0).sum() / th.size
        logging.debug(f"Epoch {epoch + 1} - Threshold={threshold}. Got signal ratio {signal_ratio}")
        diff = abs(signal_ratio - signal_ratio_target)

        if diff <= signal_ratio_diff:
            return th
        if len(set(trend)) == 2:
            rate *= 0.8
        if signal_ratio > signal_ratio_target:
            threshold += round(10 * rate)
            trend.append("+")
        elif signal_ratio < signal_ratio_target:
            threshold -= round(10 * rate)
            trend.append("-")

    logging.warning(f"Cannot found proper threshold parameter. Aborted.")
    sys.exit(2)


def get_background(th_img, rgb_img, size=10000):
    """
    Get pixels that are considered to be the background.

    Args:
        th_img (np.ndarray)     : A 2D numpy array of the threshold image.
        rgb_img (np.ndarray)    : A 2D numpy array of the rgb image.
        size (int)              : The sampling size.

    Returns:
        np.ndarray              : A 3D numpy array of the selected background pixels.
    """
    # Get the coordinates of those with value 0 on thresholded image
    row_idx, col_idx = np.where(th_img == 0)
    # Random select 10000 pixels as reference
    selected_idx = np.random.randint(low=0, high=len(row_idx), size=size)
    background_pixels = np.empty((0, rgb_img.shape[-1]))
    # Get the matrix from RGB image
    for idx in selected_idx:
        background_pixels = np.vstack((background_pixels, rgb_img[row_idx[idx], col_idx[idx]]))
    return background_pixels


def color_balance_RGBscaling(img, background_pixels):
    """
    Use background pixels to do color balance. Adjust the background to gray(60, 60, 60) and apply the offset to other pixels.

    Args:
        img (np.ndarray)                : A numpy array of the image.
        background_pixels (np.ndarray)  : A 3D numpy array of the selected background pixels or a region on a rgb image.

    Returns:
        np.ndarray                      : A 3D numpy array of the adjusted rgb image.
    """
    if len(background_pixels.shape) == 3:
        background_mat_mean = background_pixels.mean(axis=0).mean(axis=0)
    elif len(background_pixels.shape) == 2:
        background_mat_mean = background_pixels.mean(axis=0)
    scaling_array = np.divide(np.array([60, 60, 60]), background_mat_mean)

    arr_2d = img.reshape(-1, img.shape[-1])
    # np.diag turns np.array([45,45,45]) -> np.array([[45,0,0],[0,45,0],[0,0,45]])
    result_2d = np.dot(arr_2d, np.diag(scaling_array)).round().astype(int)

    # Reshape the result back to the shape of the original matrix
    result = result_2d.reshape(img.shape)

    # Limit the maximum value to 255
    result = result.clip(max=255)

    return result


def find_proper_iterations_for_dilations(gray_img, area_target=0.22, iterations=10, epochs=10, **kwargs):
    """
    Find a proper iterations for dilations to have a plates with correct size. Failed to find a proper param will terminate the
    program.

    Args:
        gray_img (np.ndarray)   : A 2D numpy array of the grayscaled image.
        area_target (float)     : The target of area ratio (area of the largest plate / total image) of the plate. (default=0.22)
        iterations (int)        : Initial setting of the iteration. (default=10)
        epochs (int)            : The maximum round of searching. (default=10)
        (get_img) (bool)        : (Optional) Return the 2D numpy array of the final image.

    Returns:
        list                    : A sorted list of contours found in the image.

        np.ndarray|None         : Return a 2D numpy array if get_img=True. Otherwise return None.
    """
    logging.debug(f"Finding an iterations that the biggest contour area ratio approximate to {area_target} ...")
    trend = deque(maxlen=2)
    rate = 1
    for epoch in range(epochs):
        sorted_contours, image = dilation_and_findcontours(gray_img, iterations=iterations, **kwargs)
        max_contour = sorted_contours[0]
        x, y, w, h = cv.boundingRect(max_contour)
        area_ratio = (w * h) / gray_img.size
        logging.debug(
            f"Epoch {epoch + 1} - Dilate with iterations={iterations}. The biggest contour area ratio is {area_ratio}"
        )
        diff = abs(area_ratio - area_target)
        if diff <= 0.02:
            return sorted_contours, image
        if len(set(trend)) == 2:
            rate *= 0.5
        if area_ratio < area_target:
            iterations += round(diff * 200 * rate)
            trend.append("+")
        elif area_ratio > area_target:
            iterations -= round(diff * 200 * rate)
            trend.append("-")
        if iterations <= 0:
            break

    logging.warning(f"Cannot found proper dilation parameter. Aborted.")
    sys.exit(2)


def find_plates(rgb_img, sorted_contours, n=3, area_ratio_h=0.30, area_ratio_l=0.15, **kwargs):
    """
    Find the largest n contours that have correct size. Failed to find any will terminate the program.

    Args:
        rgb_img (np.ndarray)    : A numpy array of the image.
        sorted_contours (list)  : A 3D numpy array of the selected background pixels or a region on a rgb image.
        n (int)                 : The number of plate expected to be found.
        area_ratio_h (float)    : The higest ratio of an area considered to be a plate.
        area_ratio_l (float)    : The lowest ratio of an area considered to be a plate.
        (output_dir) (str)      : (Optional) The directory of the output path.
        (file) (pathlib.Path)   : (Optional) The file basename of the output.

    Returns:
        None
    """
    plate_number = 1
    logging.debug(f"Finding for plate that the ratio falls between {area_ratio_l}-{area_ratio_h} ...")
    for i in range(min(len(sorted_contours), n)):
        contour = sorted_contours[i]
        x, y, w, h = cv.boundingRect(contour)
        area_ratio = (w * h) / (rgb_img.shape[0] * rgb_img.shape[1])
        logging.debug(f"Found plate with ratio {area_ratio}")
        if area_ratio >= area_ratio_l and area_ratio <= area_ratio_h:
            crop = rgb_img[y : y + h, x : x + w]
            if all(i in kwargs for i in ("output_dir", "file")):
                output_dir = kwargs.get("output_dir")
                file = kwargs.get("file")
                output = f"{output_dir}/{file.stem}_{plate_number}{file.suffix}"
                cv.imwrite(output, crop)
                logging.info(f"Write plate{plate_number} image to {output}")
            plate_number += 1

    if plate_number == 1:
        logging.warning(f"Cannot found any contour which area falls between {area_ratio_l}-{area_ratio_h}.")
        sys.exit(2)


def main():
    logging.basicConfig(format="%(asctime)s autocrop %(levelname)s [%(filename)s] %(message)s", level="INFO")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(prog=main.__name__)

    parser.add_argument("image", help="Image that contains colony plates.")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("-n", "--n_plates", type=int, default=3, help="Number of plates on the petri dish. (Default=3)")
    parser.add_argument(
        "-p",
        "--param",
        choices=["H", "M", "L"],
        help="Force to use parameters for high/medium/low growing plate.",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("-q", "--quiet", action="store_true", help="Quiet mode to suppress msg.")
    mode.add_argument("-v", "--verbose", action="store_true", help="Verbose mode for debug.")

    args = parser.parse_args()

    file, output_dir = Path(args.image), Path(args.outdir)
    output_dir.mkdir(exist_ok=True)

    # Put file name into the logger.
    f = ContextFilter()
    f.add_filename(file.name)
    logger.addFilter(f)

    if args.verbose:
        logger.setLevel("DEBUG")
        for handler in logger.handlers:
            handler.setLevel("DEBUG")
    elif args.quiet:
        logger.setLevel("WARNING")
        for handler in logger.handlers:
            handler.setLevel("WARNING")

    # Check if the processed files already exist.
    for suffix in ["M", "1", "2", "3"]:
        filepath = output_dir / f"{file.stem}_{suffix}{file.suffix}"
        if filepath.exists():
            logging.warning("Processed files already exist. Aborted.")
            sys.exit(1)

    logging.info(f"Start processing {file}")

    # Load image and convert it to grayscale.
    imageRGB = cv.imread(str(file), cv.COLOR_BGR2RGB)

    # Scale the images to a specific resolution and fix the distortion.
    imageRGB = scale_image(imageRGB, adj_width=4000, distortion_offset=0.95)

    gray = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    hsv = cv.cvtColor(imageRGB, cv.COLOR_RGB2HSV)

    # Downscales the grayscale image if it is too large to speed up largest inner rectangle detection.
    if gray.shape[1] > 1000:
        gray, scaling_factor = scale_image(gray, adj_width=1000, get_scal_f=True)
        hsv = cv.resize(hsv, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_AREA)

    # Fill the container to improve the largest inner rectangle detection.
    filled_gray = fill_container(gray)

    # Find the largest inner rectangle, which is the square dish.
    try:
        logging.debug("Detecting container...")
        x, y, w, h = detect_largest_inner_rectangle(filled_gray)
        logging.debug("Got image that contains container only.")
    except:
        logging.error("Timeout. Cannot found largest interior rectangle.")
        sys.exit(1)

    # Get lower region that contains markers. The values were obtained empirically.
    lower_bound = np.array([0, 74, 190])
    upper_bound = np.array([150, 210, 255])

    # Detect markers. Only search on part of the image below the container.
    hsv = hsv[y + h :, x : x + w]
    mx, my, mw, mh = get_markers(hsv, lower_bound, upper_bound)

    # If the image have been downscaled,
    # upscale the coordinates and dimensions back to the original image.
    if "scaling_factor" in locals():
        logging.debug("Upscale the coordinates and dimensions back to apply on the original image.")
        x, y, w, h = [int(i / scaling_factor) for i in [x, y, w, h]]
        mx, my, mw, mh = [int(i / scaling_factor) for i in [mx, my, mw, mh]]

    # Get the dish container and markers.
    image_container = imageRGB[y : y + h, x : x + w]
    markers_region = imageRGB[y + h :, x : x + w]
    marker = markers_region[my : my + mh, mx : mx + mw]

    # Shrink the image by 4% to remove the container's edges.
    image_container = shrink_image(image_container, perc=4)

    # The second part of the pargram. Find and crop out the 96-well plates.
    logging.debug("Looking for plates in the container image.")

    # Increase contrast.
    blur = cv.GaussianBlur(image_container, (5, 5), 0)
    enhanced_img = increase_contrast(blur)
    gray = cv.cvtColor(enhanced_img, cv.COLOR_RGB2GRAY)

    # Color balance by the background and output to file.
    _, th = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    signal_ratio = (th > 0).sum() / th.size

    background_reference = get_background(th, image_container)
    imageRGB = color_balance_RGBscaling(imageRGB, background_reference)

    # Parameters for high/medium/low growing plates.
    param_dict = {
        "H": {"signal_ratio_target": 0.20, "signal_ratio_diff": 0.05},
        "M": {"signal_ratio_target": 0.11, "signal_ratio_diff": 0.04},
        "L": {"signal_ratio_target": 0.04, "signal_ratio_diff": 0.03},
    }

    # If args.param is set, use the corresponding parameter set.
    # Else, use the signal ratio obtained above to choose a parameter set automatically.
    if args.param:
        select = args.param
    else:
        if signal_ratio < param_dict["M"]["signal_ratio_target"] - param_dict["M"]["signal_ratio_diff"]:
            select = "L"
        elif signal_ratio > param_dict["M"]["signal_ratio_target"] + param_dict["M"]["signal_ratio_diff"]:
            select = "H"
        else:
            select = "M"

    logging.info(f'Use "{select}" parameter set')

    output = f"{output_dir}/{file.name}"
    cv.imwrite(output, imageRGB)
    logging.info(f"Write adjusted image to {output}")

    # Find the proper threshold.
    th = find_proper_threshold(gray, **param_dict[select])

    # Use opening to denoise.
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=5)

    # Find the proper iterations.
    sorted_contours, _ = find_proper_iterations_for_dilations(opening)

    # Find, crop and output the first n largest contours.
    image_container = color_balance_RGBscaling(image_container, background_reference)
    find_plates(image_container, sorted_contours, n=args.n_plates, output_dir=output_dir, file=file)

    # Output the marker if at least one plates being cut out.
    marker = color_balance_RGBscaling(marker, background_reference)
    marker_path = f"{output_dir}/{file.stem}_M{file.suffix}"
    cv.imwrite(marker_path, marker)
    logging.info(f"Write marker image to {marker_path}")


main.__name__ = "autocrop.py"

if __name__ == "__main__":
    main()
