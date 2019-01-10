import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage import exposure
from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

# Read the images to be aligned
im1_color = cv2.imread("./progenitors 2/p3_1_v1_gray.png");
im2_color = cv2.imread("./replicates_p4/p4_1_v1.png");

# Convert images to grayscale
im1 = cv2.cvtColor(im1_color,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2_color,cv2.COLOR_BGR2GRAY)

im1_sq = im1[300:980,300:980]
im2_sq = im2[300:980,300:980]

# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_EUCLIDEAN

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 5000;

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-9;

# Define termination criteriaeM
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_sq,im2_sq,warp_matrix, warp_mode, criteria)

if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);


overlay = im1.copy()
cv2.addWeighted(im2_aligned, 0.5, overlay, 0.5, 0, overlay)

'''
# Show final results
cv2.imshow("Image 1", im1)
cv2.imshow("Image 2", im2)
cv2.imshow("Aligned Image 2", overlay)
cv2.waitKey(25)
'''

#identify overlappying colonies
class HoughPlateDetect:
    
        def __init__(self, image, minRadii, maxRadii):
            self.image = image    
            self.edges = canny(image, sigma=3, low_threshold = 10, high_threshold = 50)
            self.tryRadii= np.arange(minRadii, maxRadii, 2)
            self.centers = []            #centers of circles from hough
            self.accums = []            #values of hough at centers
            self.radii = []                #radii of test radii that match hough
    
        def perform_transform(self, peaks_to_lookfor):
            hough_transform = hough_circle(self.edges, self.tryRadii)
    
            for triedRadii, houghResult in zip(self.tryRadii, hough_transform):
                num_peaks = peaks_to_lookfor
                peaks = peak_local_max(houghResult, num_peaks = num_peaks) #returns list of coordinates of peaks (x,y)
    
                # add higest accum positions to centers, and the values to accums
                self.centers.extend(peaks)
                self.accums.extend(houghResult[peaks[:, 0], peaks[:,1]])
                self.radii.extend([triedRadii]* num_peaks)
            
        def return_accums(self):
            return self.accums
        
        def return_centers(self): 
            return self.centers
    
        def return_visualization_parameters(self):
            #return parameters for best circle properties (center position and radius)
            best_accum_index = np.argsort(self.accums)[::-1][0]
            center_x, center_y = self.centers[best_accum_index]
            radius = self.radii[best_accum_index]
    
            return [center_y, center_x, radius]
    
        def visualize_hough(self, numCircles):
            hough = color.gray2rgb(self.image)
            
            #plot numCircles circles for hai(highest accumulant index)
            for hai in np.argsort(self.accums)[::-1][:numCircles]:
                
                center_x, center_y = self.centers[hai]
            radius = self.radii[hai]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            hough[cy, cx] = (220, 20, 20)
    
            return hough

class OtsuSegment:

    def __init__(self, some_image, map):
        self.image = some_image
        self.map = map
        self.val = filters.threshold_otsu(self.map)

    def visualize_otsu(self):    
        
        hist, bins_center = exposure.histogram(self.image)

        plt.figure(figsize=(9, 4)) #wh in inches on screen
        plt.subplot(121)
        plt.imshow(self.image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(self.image > self.val, cmap='gray', interpolation='nearest')
        plt.axis('off')
        #plt.subplot(133)
        #plt.plot(bins_center, hist, lw=2)
        #plt.axvline(self.val, color='k', ls='--')

        plt.tight_layout()
        plt.show()

    def return_otsu(self):
        return self.image > self.val
        

class CircleMask:
    
    def __init__(self, image, parameter, pixelbuffer):
        self.image = image
        self.ccx = parameter[0]
        self.ccy = parameter[1]
        self.cradius = parameter[2]
        self.buffer = pixelbuffer

        ly,lx = image.shape
        y,x  = np.ogrid[0:ly , 0 :lx]
        
        mask = (x- self.ccx)**2 + (y - self.ccy)**2 > (self.cradius - self.buffer)**2
        self.image[mask] = 0
        
    def return_mask(self):
        return self.image
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#colony annotation section
if __name__ == '__main__':
    
    hough1 = HoughPlateDetect(im1, 500, 550)
    hough2 = HoughPlateDetect(im2_aligned, 500, 550)
    
    #image = img_as_ubyte(img)

   # greyscale_im = GreyScaleConverter(image).return_greyscale(False)
    #print "finished greyscale"
    #print hough.return_accums()
    
    ''' hough transform section'''

    #get plate outline through hough transform
    #hough = HoughPlateDetect(image, 2050, 2080)
    hough1.perform_transform(2)
    hough2.perform_transform(2)

    #get size of this hough transform outline
    parameters1 = hough1.return_visualization_parameters()
    parameters2 = hough2.return_visualization_parameters()
    #create sample square image of plate to be used in thresholding
    cen_x1 = parameters1[0]
    cen_y1 = parameters1[1]
    radius1 = parameters1[2]
    print(radius1)

    leftmost_col1 = int(cen_x1 - float(radius1)/math.sqrt(2))
    rightmost_col1 = int(cen_x1 + float(radius1)/math.sqrt(2))
    upmost_row1 = int(cen_y1 + float(radius1)/math.sqrt(2))
    lowest_row1 = int(cen_y1 - float(radius1)/math.sqrt(2))

    square1 = []
    for i in range(lowest_row1-1, upmost_row1):
        square1.append(im1[i][leftmost_col1 -1:rightmost_col1 -1])
    
    cen_x2 = parameters2[0]
    cen_y2 = parameters2[1]
    radius2 = parameters2[2]
    print(radius2)
    
    leftmost_col2 = int(cen_x2 - float(radius2)/math.sqrt(2))
    rightmost_col2 = int(cen_x2 + float(radius2)/math.sqrt(2))
    upmost_row2 = int(cen_y2 + float(radius2)/math.sqrt(2))
    lowest_row2 = int(cen_y2 - float(radius2)/math.sqrt(2))

    square2 = []
    for i in range(lowest_row2-1, upmost_row2):
        square2.append(im2_aligned[i][leftmost_col2 -1:rightmost_col2 -1])

    #optionally view hough result
    #hough_img_overlay = hough.visualize_hough(3)
    #plt.imshow(hough_img_overlay, cmap = plt.cm.gray)
    #plt.show()
    #plt.imshow(square, cmap = plt.cm.gray)
    #plt.show()
    otsu1 = OtsuSegment(im1, np.asarray(square1))
    otsu2 = OtsuSegment(im2_aligned, np.asarray(square2))
    #otsu.visualize_otsu()

    #plt.imshow(otsu.return_otsu(), cmap = plt.cm.gray)
    #plt.show()
#########################################################################################
    '''circle mask testing, and perimeter elimination'''

    #labeling segmented region
    labels1, numlabels1 = measure.label(otsu1.return_otsu(), background = 0, return_num = True)
    labels2, numlabels2 = measure.label(otsu2.return_otsu(), background = 0, return_num = True)
    
    #count sizes of each from flattened array
    initial_sizes1 = np.bincount(labels1.ravel())
    initial_sizes1[0] = 0
    
    initial_sizes2 = np.bincount(labels2.ravel())
    initial_sizes2[0] = 0

    #setting foreground regions < size(100) to background
    small_sizes1 = initial_sizes1 < 100
    small_sizes1[0] = 0 
    small_sizes2 = initial_sizes2 < 100
    small_sizes2[0] = 0

    #get rid of large foreground objects
    large_sizes1 = initial_sizes1 > 9500
    large_sizes1[0] = 0
    large_sizes2 = initial_sizes2 > 9500
    large_sizes2[0] = 0

    labels1[small_sizes1[labels1]] = 0
    labels1[large_sizes1[labels1]] = 0
    labels2[small_sizes2[labels2]] = 0
    labels2[large_sizes2[labels2]] = 0

    preprocessed_sizes1 = np.bincount(labels1.ravel())
    preprocessed_sizes2 = np.bincount(labels2.ravel())

    #CM = CircleMask(otsu.return_otsu(), parameters, 300)
    #plt.imshow(CM.return_mask(), cmap = plt.cm.gray)
    #plt.show()
    #
    # plt.imshow(labels, cmap = plt.cm.gray)
    # plt.show()

    #apply circle mask to labels grid
    circle_mask1 = CircleMask(labels1, parameters1, 100)
    labels_mask1 = circle_mask1.return_mask()
    mask_sizes1 = np.bincount(labels_mask1.ravel())
    mask_sizes1[0] = 0
    
    circle_mask2 = CircleMask(labels2, parameters2, 100)
    labels_mask2 = circle_mask2.return_mask()
    mask_sizes2 = np.bincount(labels_mask2.ravel())
    mask_sizes2[0] = 0

    #print (str(np.count_nonzero(mask_sizes)) + " labels before hough mask elimination. \n")
    #getting rid of perimeter colonies
    #binwidth = 5
    #plt.hist(mask_sizes, bins=range(5, max(mask_sizes) + binwidth, binwidth))
    
    #plt.plot(mask_sizes)
    #plt.show()

    #loop through to see which bin counts are truncated (brute force)
    bins_truncated1 = []
    for bin in range(len(mask_sizes1)):
        if mask_sizes1[bin] < preprocessed_sizes1[bin]:
            bins_truncated1.append(bin)
    
    #reset masked label map to remove truncated bins
    for row in range(len(labels_mask1)):
        for col in range(len(labels_mask1[0])):
            if labels_mask1[row][col] in bins_truncated1:
                labels_mask1[row][col] = 0
    
    bins_truncated2 = []
    for bin in range(len(mask_sizes2)):
        if mask_sizes2[bin] < preprocessed_sizes2[bin]:
            bins_truncated2.append(bin)
    
    #reset masked label map to remove truncated bins
    for row in range(len(labels_mask2)):
        for col in range(len(labels_mask2[0])):
            if labels_mask2[row][col] in bins_truncated2:
                labels_mask2[row][col] = 0

    hough_sizes1 = np.bincount(labels_mask1.ravel())
    hough_sizes1[0] = 0
    
    hough_sizes2 = np.bincount(labels_mask2.ravel())
    hough_sizes2[0] = 0

    #print (str(np.count_nonzero(hough_sizes)) + " after hough mask elimination. \n")

    plt.imshow(labels_mask1, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(labels_mask2, cmap=plt.cm.gray)
    plt.show()
    
    petites = 0
    total = len(hough_sizes1) - 1
    
    for label in np.unique(labels_mask1):
        if label != 0:
            xs, ys = np.where(labels_mask1 == label)
            area = 0
            matching = 0
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                if labels_mask2[x][y] != 0:
                    matching = matching + 1
                area = area + 1
            if matching/area < 0.2:
                petites = petites + 0
    
    print("The number of petites is", petites)
