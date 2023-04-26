import os
import argparse
# import numpy as np
import cv2
# import CoOrdinatesModel
import xml.etree.cElementTree as ET

from utils import *
from background_marker import *
font = cv2.FONT_HERSHEY_SIMPLEX


def generate_background_marker(file):
    """
    Generate background marker for an image

    Args:
        file (string): full path of an image file

    Returns:
        tuple[0] (ndarray of an image): original image
        tuple[1] (ndarray size of an image): background marker
    """

    # check file name validity
    if not os.path.isfile(file):
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    # update marker to remove blues
    # remove_blues(original_image, marker)

    return original_image, marker


class CoOrdinatesModel:
    def __init__(self, xmi, ymi, xma, yma):
        self.xmi = xmi
        self.ymi = ymi
        self.xma = xma
        self.yma = yma


def segment_leaf(image_file, filling_mode, smooth_boundary, marker_intensity):
    """
    Segments leaf from an image file

    Args:
        image_file (string): full path of an image file
        filling_mode (string {no, flood, threshold, morph,area}):
            how holes should be filled in segmented leaf
        smooth_boundary (boolean): should leaf boundary smoothed or not
        marker_intensity (int in rgb_range): should output background marker based
                                             on this intensity value as foreground value

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
                            or the segmented image based on marker_intensity value
    """
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    largest_mask, contours, hierarchy, area = \
        select_largest_obj(bin_image, fill_mode=filling_mode,
                           smooth_boundary=smooth_boundary)

    image_copy = original.copy()
    with_contours = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                                     lineType=cv2.LINE_AA)

    # print(f'Countour Max Area {cv2.contourArea(max(contours, key=cv2.contourArea))}')

    max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
    listLocations = []

    for cont, contor in enumerate(contours):
        are = cv2.contourArea(contor)
        if are == max_area:
            max_contour_loc = cont
            break

    # Draw a bounding box around all contours
    for count,c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        # Ignore contours that are very thin (like edges)
        if w > 10 and h > 10:
            # contour_damage = (np.count_nonzero(c) / area['TotalArea']) * 100
            # print(f'contour Damage {contour_damage}')
            # Make sure contour area is large enough
            if (cv2.contourArea(c)) > 30:
                cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)
                # cv2.imshow('cutted contour', with_contours[y:y + h, x:x + w])
                # print('Average color (BGR): ', np.array(cv2.mean(with_contours[y:y + h, x:x + w])).astype(np.uint8))
                # cv2.waitKey(0)

                if hierarchy[0, count, 3] == max_contour_loc:
                    cv2.putText(with_contours, 'internal_damage', (x, y - 5), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
                    listLocations.append( CoOrdinatesModel(x, y, w+x, h+y) ) #wrong formula
                    area['location'].append(CoOrdinatesModel(x, y, h+x, w+y))
                # print(f'count is {max_contour_loc} and area is {area}')


    # Find biggest contour
    biggestContour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggestContour)

    # draw the biggest contour (c) in green
    cv2.rectangle(with_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # cv2.putText(with_contours, findClass(area["DamagePercent"]), (x, y - 5), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

    # cv2.imwrite('D:/IMG_CONTOURS', with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if marker_intensity > 0:
        largest_mask[largest_mask != 0] = marker_intensity
        image = largest_mask
    else:
        # apply marker to original image
        image = original.copy()
        image[largest_mask == 0] = np.array([0, 0, 0])

    return original, image, area, with_contours


def rgb_range(arg):
    """
    Check if arg is in range for rgb value(between 0 and 255)

    Args:
        arg (int convertible): value to be checked for validity of range

    Returns:
        arg in int form if valid

    Raises:
        argparse.ArgumentTypeError: if value can not be integer or not in valid range
    """

    try:
        value = int(arg)
    except ValueError as err:
       raise argparse.ArgumentTypeError(str(err))

    if value < 0 or value > 255:
        message = "Expected 0 <= value <= 255, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value


if __name__ == '__main__':
    # handle command line arguments
    parser = argparse.ArgumentParser('segment')
    parser.add_argument('-m', '--marker_intensity', type=rgb_range, default=0,
                        help='Output image will be as black background and foreground '
                             'with integer value specified here')
    parser.add_argument('-f', '--fill', choices=['no', 'flood', 'threshold', 'morph', 'area'],
                        help='Change hole filling technique for holes appearing in segmented output',
                        default='flood')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='Output image with smooth edges')
    parser.add_argument('-d', '--destination',
                        help='Destination directory for output image. '
                             'If not specified destination directory will be input image directory')
    parser.add_argument('-o', '--with_original', action='store_true',
                        help='Segmented output will be appended horizontally to the original image')
    parser.add_argument('-i', '--image_source', help='A path of image filename or folder containing images')
    parser.add_argument('-c', '--contours', help='Directory to store images with contours')
    parser.add_argument('-g', '--image_segmentation', help='Directory to store segmented images')
    
    # set up command line arguments conveniently
    args = parser.parse_args()
    filling_mode = FILL[args.fill.upper()]
    smooth = True if args.smooth else False
    if args.destination:
        if not os.path.isdir(args.destination):
            print(args.destination, ': is not a directory')
            exit()

    # set up files to be segmented and destination place for segmented output
    if os.path.isdir(args.image_source):
        files = [entry for entry in os.listdir(args.image_source)
                 if os.path.isfile(os.path.join(args.image_source, entry))]
        base_folder = args.image_source

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            if args.image_source.endswith(os.path.sep):
                args.image_source = args.image_source[:-1]
            destination = args.image_source + '_markers'
            os.makedirs(destination, exist_ok=True)
    else:
        folder, file = os.path.split(args.image_source)
        files = [file]
        base_folder = folder

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            destination = folder

    for file in files:
        try:
            # read image and segment leaf
            original, output_image, AREA, withContours = \
                segment_leaf(os.path.join(base_folder, file), filling_mode, smooth, args.marker_intensity)
            h, w, c = output_image.shape

        except ValueError as err:
            if str(err) == IMAGE_NOT_READ:
                print('Error: Could not read image file: ', file)
            elif str(err) == NOT_COLOR_IMAGE:
                print('Error: Not color image file: ', file)
            else:
                raise
        # if no error when segmenting write segmented output
        else:
            # handle destination folder and fileaname
            filename, ext = os.path.splitext(file)
            if args.with_original:
                new_filename = filename + '_marked_merged' + ext
            else:
                new_filename = filename + ext
            new_xml_file = filename + '.xml'
            new_xml_file = os.path.join(destination, new_xml_file)
            new_filename = os.path.join(destination, new_filename)


            # change grayscale image to color image format i.e need 3 channels
            if args.marker_intensity > 0:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)

            # write the output
            if args.with_original:
                cv2.imwrite(new_filename, np.hstack((original, output_image)))
            else:
                cv2.imwrite(new_filename, original)

            if args.contours:
                new_contour_filename = os.path.join(args.contours, filename + ext)
                cv2.imwrite(new_contour_filename, withContours)
            else:
                new_contour_filename = os.path.join(folder, filename + '_contour' +ext)
                cv2.imwrite(new_contour_filename, withContours)

            if args.image_segmentation:
                new_segmented_directory = os.path.join(args.image_segmentation, filename + ext)
                cv2.imwrite(new_segmented_directory, output_image)
            else:
                new_segmented_directory = os.path.join(folder, filename + '_segmented' + ext)
                cv2.imwrite(new_segmented_directory, output_image)

            print('Marker generated for image file: ', file)

            # write xml file data
            root = ET.Element("annotation")

            ET.SubElement(root, "folder").text = "XML"
            ET.SubElement(root, "filename").text = filename + ext
            ET.SubElement(root, "path").text = destination
            ET.SubElement(root, "class").text = findClass(AREA["DamagePercent"])
            ET.SubElement(root, "Damage").text = "{:.2f}".format(AREA["DamagePercent"])
            source = ET.SubElement(root, "size")
            ET.SubElement(source, "width").text = str(w)
            ET.SubElement(source, "height").text = str(h)
            ET.SubElement(source, "depth").text = str(c)
            for parameters in AREA["location"]:
                imgObject = ET.SubElement(root, "object")
                difficult = ET.SubElement(imgObject, "difficult").text = "0"
                pClass = ET.SubElement(imgObject, "bndbox")
                usr4 = ET.SubElement(imgObject, "name")
                usr4.text = "Internal Damage"
                usr0 = ET.SubElement(pClass, "xmin")
                usr0.text = str(parameters.xmi)
                usr1 = ET.SubElement(pClass, "ymin")
                usr1.text = str(parameters.ymi)
                usr2 = ET.SubElement(pClass, "xmax")
                usr2.text = str(parameters.xma)
                usr3 = ET.SubElement(pClass, "ymax")
                usr3.text = str(parameters.yma)

            tree = ET.ElementTree(root)

            # xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")

            with open(new_xml_file, "wb") as file:
                tree.write(file)
                file.close()
