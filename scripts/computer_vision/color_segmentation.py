import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
    """
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

    bounding_box = ((0,0),(0,0))
    
	# Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set the lower and upper HSV limits for the orange color (started with 5, 100, 100 and 20, 255, 255)
    # lower_orange = np.array([5, 100, 100])
    # upper_orange = np.array([20, 255, 255])
	# These values failed test 9, 14, 15

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([18, 255, 255])
    # Create a mask to keep only the orange pixels
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Erode the mask to remove noise
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=2)

    # Dilate the mask to improve the shape
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)

    # Find the contours in the dilated mask
    contours = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # Find the bounding box of the largest contour
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            bounding_box = ((x, y), (x + w, y + h))


	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
    return bounding_box