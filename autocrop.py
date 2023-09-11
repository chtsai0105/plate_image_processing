#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import largestinteriorrectangle as lir
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


def find_contours(grid):
    contours = cv.findContours(grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


@timeout(120)
def detect_largest_rectangle(grid):
    contours = find_contours(grid)
    largest_contour = max(contours, key=cv.contourArea)

    epsilon = 0.01 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)
    d1, d2, d3 = approx.shape

    x, y, w, h = lir.lir(approx.reshape((d2, d1, d3)))
    return x, y, w, h


def dilation_and_findcontours(grid, iterations=30):
    kernel = np.ones((9, 9), np.uint8)
    dilation = cv.dilate(grid, kernel, iterations=iterations)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations=10)

    contours = find_contours(closing)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

    return sorted_contours


def cleanup_marker(output_dir, filename, ext):
    removeList = []
    for suffix in ["M", "1", "2", "3"]:
        filepath = output_dir / f"{filename}_{suffix}{ext}"
        removeList.append(str(filepath))
        filepath.unlink(missing_ok=True)
    logging.info("Remove already generated files.")
    sys.exit(2)


def main():
    logging.basicConfig(format="%(asctime)s autocrop %(levelname)s [%(filename)s] %(message)s", level="INFO")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(prog=main.__name__)

    parser.add_argument("image", help="Image that contains colony plates.")
    parser.add_argument("-o", "--outdir", type=str, help="Output directory.")
    parser.add_argument("-H", "--high", action="store_true", help="Use parameters for high growth plate")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("-q", "--quiet", action="store_true", help="Quiet mode to suppress msg.")
    mode.add_argument("-v", "--verbose", action="store_true", help="Verbose mode for debug.")

    args = parser.parse_args()

    file, output_dir = Path(args.image), Path(args.outdir)
    output_dir.mkdir(exist_ok=True)

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

    # Check if processed files already exist
    for suffix in ["M", "1", "2", "3"]:
        filepath = output_dir / f"{file.stem}_{suffix}{file.suffix}"
        if filepath.exists():
            logging.warning("Processed files already exist. Aborted.")
            sys.exit(1)

    logging.info(f"Start processing {file}")

    # Load image and convert it to grayscale
    imageRGB = cv.imread(str(file), cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    hsv = cv.cvtColor(imageRGB, cv.COLOR_RGB2HSV)

    # Down scale the grayscale image if it is too large to speed up largest rectangle detection
    if gray.shape[1] > 2000:
        adj_width = 2000
        scaling_factor = adj_width / gray.shape[1]
        adj_height = int(gray.shape[0] * scaling_factor)
        scaled_dim = (adj_width, adj_height)
        gray = cv.resize(gray, scaled_dim, interpolation=cv.INTER_AREA)
        hsv = cv.resize(hsv, scaled_dim, interpolation=cv.INTER_AREA)

    # Preprocessing
    _, th1 = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    _, th2 = cv.threshold(gray, 50, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    overlap = cv.bitwise_or(th1, th2)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(overlap, cv.MORPH_OPEN, kernel, iterations=3)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=5)
    contours = find_contours(closing)
    closing = cv.drawContours(closing, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

    # Find the largest rectangle, which is the square dish
    try:
        logging.debug("Detecting container...")
        x, y, w, h = detect_largest_rectangle(closing)
        logging.debug("Got image that contains container only.")
    except:
        logging.error("Timeout. Cannot found largest interior rectangle.")
        sys.exit(1)

    # Get lower region that contains markers
    lower_bound = np.array([0, 74, 190])
    upper_bound = np.array([150, 210, 255])
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    _, th = cv.threshold(mask, 90, 255, cv.THRESH_BINARY)
    markers_region = th[y + h :, x : x + w]
    kernel = np.ones((5, 5), np.uint8)
    markers_region = cv.morphologyEx(markers_region, cv.MORPH_OPEN, kernel, iterations=1)
    markers_region = cv.dilate(markers_region, kernel, iterations=2)
    markers_region = cv.morphologyEx(markers_region, cv.MORPH_CLOSE, kernel, iterations=5)

    logging.debug("Detecting markers...")
    contours = find_contours(markers_region)
    max_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
    mx, my, mw, mh = cv.boundingRect(max_contour)
    logging.info("Got image that contains markers only.")

    # If the image have been downscaled,
    # up scale the coordinates and dimensions back to apply on the original image
    if "scaling_factor" in locals():
        logging.debug("Upscale the coordinates and dimensions back to apply on the original image.")
        x, y, w, h = [int(i / scaling_factor) for i in [x, y, w, h]]
        mx, my, mw, mh = [int(i / scaling_factor) for i in [mx, my, mw, mh]]

    # Get the dish container and markers
    image_container = imageRGB[y : y + h, x : x + w]
    markers_region = imageRGB[y + h :, x : x + w]
    marker = markers_region[my : my + mh, mx : mx + mw]

    # Shrink the image to remove the container's edges
    h, w, _ = image_container.shape
    h = int(h * 0.02)
    w = int(w * 0.02)
    image_container = image_container[h:-h, w:-w]

    marker_path = f"{output_dir}/{file.stem}_M{file.suffix}"
    cv.imwrite(marker_path, marker)
    logging.info(f"Write marker image to {marker_path}")

    # Get plates
    logging.info("Looking for plates in the container image")

    # Increase contrast
    lab = cv.cvtColor(image_container, cv.COLOR_RGB2LAB)
    l_channel, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv.merge((cl, a, b))
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2RGB)
    gray = cv.cvtColor(enhanced_img, cv.COLOR_RGB2GRAY)

    if args.high:
        threshold, signal_ratio_h, area_ratio_h = 100, 0.25, 0.25
    else:
        threshold, signal_ratio_h, area_ratio_h = 100, 0.06, 0.20

    # Find the proper threshold
    signal_ratio = 0
    trend = ""
    while signal_ratio > signal_ratio_h or signal_ratio < 0.01:
        _, th = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
        signal_ratio = (th > 0).sum() / th.size
        logging.debug(f"Threshold={threshold}. Got signal ratio {signal_ratio}")

        if signal_ratio > signal_ratio_h:
            threshold += 10
            trend += "+"
        elif signal_ratio < 0.01:
            threshold -= 10
            trend += "-"

        if len(set(trend)) == 2 or len(trend) > 10:
            logging.warning(f"[{file}] Cannot found proper threshold parameter. Aborted.")
            cleanup_marker(output_dir, file.stem, file.suffix)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=8)

    # Find the proper iterations
    iterations = 30
    area_ratio = 0
    trend = ""
    while area_ratio < 0.10 or area_ratio > area_ratio_h:
        sorted_contours = dilation_and_findcontours(opening, iterations=iterations)
        max_contour = sorted_contours[0]
        x, y, w, h = cv.boundingRect(max_contour)
        area_ratio = (w * h) / gray.size
        logging.debug(f"Dilate with iterations={iterations}. The biggest contour area ratio is {area_ratio}")
        if area_ratio < 0.10:
            iterations += 10
            trend += "+"
        elif area_ratio > 0.20:
            iterations -= 10
            trend += "-"

        if len(set(trend)) == 2 or len(trend) > 10:
            logging.warning(f"Cannot found proper dilation parameter. Aborted.")
            cleanup_marker(output_dir, file.stem, file.suffix)

    # Find, crop and print the first 3 largest contours
    plate_number = 1
    logging.info("Finding for plate that the ratio falls between 0.1-0.2 ...")
    for i in range(min(len(sorted_contours), 3)):
        contour = sorted_contours[i]
        x, y, w, h = cv.boundingRect(contour)
        area_ratio = (w * h) / gray.size
        logging.debug(f"Found plate with ratio {area_ratio}")
        if area_ratio >= 0.15 and area_ratio <= 0.23:
            crop = image_container[y : y + h, x : x + w]
            output = f"{output_dir}/{file.stem}_{plate_number}{file.suffix}"
            cv.imwrite(output, crop)
            logging.info(f"Write plate{plate_number} image to {output}")
            plate_number += 1

    if plate_number == 1:
        logging.warning("Cannot found any contour which area falls between 0.1-0.2.")
        cleanup_marker(output_dir, file.stem, file.suffix)


main.__name__ = "autocrop.py"

if __name__ == "__main__":
    main()
