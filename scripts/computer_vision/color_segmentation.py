import cv2
import numpy as np
import pdb

def image_print(img):
    cv2.imshow("image", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
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



    opencv_major_version = int(cv2.__version__.split('.')[0])

    # Call cv2.findContours with the appropriate output format based on the major version
    if opencv_major_version >= 4:
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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



def print_image_with_bounding_box(img, bbox):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    img = cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Testing bounding box visualization on 20 cone test images

for image in range(1, 3):
    # print("~/racecar_ws/src/visual_servoing/scripts/computer_vision/test_images_cone/test" + str(image) + ".jpg")
    img = cv2.imread("./test_images_cone/test" + str(image) + ".jpg")
    img_crop = img[int(img.shape[0]*.5):int(img.shape[0]*.85),:,:]
    cv2.imshow("image", img_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(img.shape)
    # print(img)
    # cv2.imshow("image", img)
    bbox = cd_color_segmentation(img_crop, None)
    print_image_with_bounding_box(img_crop, bbox)





