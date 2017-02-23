import NoiseReduction.GaussianBlur as GaussianBlur
import EdgeProcessing.CannyEdge as CannyEdge
from EdgeProcessing.SobelEdge import SobelEdge
from Geometry.PolarSideCounter import PolarSideCounter
import EdgeProcessing.HarrisCorner as HarrisCorner
from ImgStat.KMeans import Clusters, Cluster
from Geometry.HoughCircle import HoughCircle
import ImageOperation.Crop as Crop
from Color.ColorLayer import ColorLayer
from math import pi, sqrt
class ShapeType:
    
    BLUR_KERNEL_SIZE = 3
    BLUR_STD_DEV = 2
    CANNY_SHAPE_THRESHOLDS = (10, 20)
    HARRIS_THRESHOLD = 100000
    HARRIS_STD_DEV = 1
    HARRIS_KERNEL_SIZE = 3
    MIN_CORNER_CLUSTER = 1
    PENTAGON_STAR_CLUSTER_THRESHOLD = 8
    OCTAGON_CROSS_CLUSTER_THRESHOLD = 10
    SQUARE_RECTANGLE_EIGEN_RATIO_THRESHOLD = 1.2
    HARRIS_CIRCUMFERENCE_THRESHOLD_MULTIPLIER = 0.7
    
    
    def __init__(self, shape_img_in, simple_pca_in):
        self.shape_img = shape_img_in
        self.shape_image = self.shape_img.load()
        self.shape_img = GaussianBlur.get_gaussian_filtered_bw_img(self.shape_img, self.shape_image, ShapeType.BLUR_KERNEL_SIZE, ShapeType.BLUR_STD_DEV)
        self.shape_sobel = SobelEdge(self.shape_img)
        self.shape_sobel.get_gradient_mag_img().show()
        self.canny_img = CannyEdge.get_canny_img(self.shape_sobel, ShapeType.CANNY_SHAPE_THRESHOLDS)
        self.canny_image = self.canny_img.load()
        self.init_shape_area()
        #self.crop_canny_img = Crop.get_bw_img_cropped_to_bounds(self.canny_img, self.canny_image)
        self.polar_side_counter = PolarSideCounter(self.canny_img, self.canny_image)
        self.num_polar_side_maximums = len(self.polar_side_counter.get_maximums())
        self.num_polar_side_minimums = len(self.polar_side_counter.get_minimums())
        self.simple_pca = simple_pca_in
        self.init_shape_type()
    
    def init_shape_area(self):
        shape_layer = ColorLayer(255, (1,1))
        shape_layer.set_layer_img(self.shape_img)
        self.area = shape_layer.get_area()
    
    def init_shape_type(self):
        self.canny_img.show()
        self.polar_side_counter.get_maximums_drawn_to_img().show()
        self.polar_side_counter.get_minimums_drawn_to_img().show()
        
        self.hough_circle = HoughCircle(self.canny_img, self.canny_image, (50, 200))
        print("HIGHEST RADIUS VOTE: " + str(self.hough_circle.get_highest_vote_and_radius()))
        
        highest_vote_and_radius = self.hough_circle.get_highest_vote_and_radius()
        if highest_vote_and_radius[0] > self.get_hough_circle_vote_threshold():
            self.shape_type = "Circle"
        elif highest_vote_and_radius[0] > self.get_hough_quartercircle_vote_threshold():
            self.shape_type = "Quarter-Circle"
        elif self.num_polar_side_maximums == 2:
            self.shape_type = "Semi-Circle"
        elif self.num_polar_side_maximums == 3:
            self.shape_type = "Triangle"
        elif self.num_polar_side_maximums == 6:
            self.shape_type = "Hexagon"
        elif self.num_polar_side_maximums == 7:
            self.shape_type = "Heptagon"
        else:
            self.init_harris_corners_and_cluster()
            
            if self.num_polar_side_maximums == 5:
                if self.num_harris_clusters < ShapeType.PENTAGON_STAR_CLUSTER_THRESHOLD:
                    self.shape_type = "Pentagon"
                else:
                    self.shape_type = "Star"
            elif self.num_polar_side_maximums == 8:
                if self.num_harris_clusters < ShapeType.OCTAGON_CROSS_CLUSTER_THRESHOLD:
                    self.shape_type = "Octagon"
                else:
                    self.shape_type = "Cross"
            else: 
                eigenvalues = self.simple_pca.get_eigenvalues()
                eigen_ratio = eigenvalues[0]/eigenvalues[1]
                if eigen_ratio > ShapeType.SQUARE_RECTANGLE_EIGEN_RATIO_THRESHOLD or 1.0/eigen_ratio > ShapeType.SQUARE_RECTANGLE_EIGEN_RATIO_THRESHOLD:
                    self.shape_type = "Rectangle"
                else:
                    self.shape_type = "Square"
     
    def get_hough_circle_vote_threshold(self):
        '''returns a minimum threshold to determine whether or not the shape is a circle. Is NOT perfect because
        the pixels that make up the circumference of the circle will not line up with it's perimeter calculated using 
        pixels as a unit, as pixels are square and a line traversing across a pixel could have length anywhere between 
        a width of a pixel and width of a pixel * sqrt(2). Could be mathematicized to work, but likely not easily,
        and this will probably work fine.
        
        Made to work with a Hough Circle blob radius of 0'''
        radius = self.get_radius_of_possible_circle()
        minimum_circumference = sqrt(2) * pi * radius
        return ShapeType.HARRIS_CIRCUMFERENCE_THRESHOLD_MULTIPLIER * minimum_circumference
    
    def get_hough_quartercircle_vote_threshold(self):
        radius = self.get_radius_of_possible_quartercircle()
        minimum_circumference = sqrt(2) * pi * radius / 4.0
        return ShapeType.HARRIS_CIRCUMFERENCE_THRESHOLD_MULTIPLIER * minimum_circumference
    
    def get_radius_of_possible_circle(self):
        radius = sqrt(self.area/pi)
        return radius
    
    def get_radius_of_possible_quartercircle(self):
        radius = sqrt(self.area*4.0/pi)
        return radius
    
    def init_harris_corners_and_cluster(self):
        self.harris_corners = HarrisCorner.get_corners_over_threshold(self.shape_sobel, ShapeType.HARRIS_KERNEL_SIZE, ShapeType.HARRIS_STD_DEV, ShapeType.HARRIS_THRESHOLD)
        maxes_and_mins = list(self.polar_side_counter.get_maximums())
        for i in range(0, len(self.polar_side_counter.get_minimums())):
            maxes_and_mins.append(self.polar_side_counter.get_minimums()[i])
        for i in range(0, len(maxes_and_mins)):
            maxes_and_mins[i] = Cluster(maxes_and_mins[i].get_pixel(self.polar_side_counter.get_origin()))
        self.clusters = Clusters(self.harris_corners, maxes_and_mins)
        self.clusters.fit_data_to_clusters(1, 0)
        self.remove_clusters_with_corners_under_threshold(ShapeType.MIN_CORNER_CLUSTER)
        self.num_harris_clusters = len(self.clusters)
        
    def remove_clusters_with_corners_under_threshold(self, num):
        i = 0
        while i < len(self.clusters):
            if len(self.clusters[i]) <= num:
                del self.clusters[i]
            else:
                i += 1
                
    def get_polar_side_counter(self):
        return self.polar_side_counter
    
    def __repr__(self):
        return self.shape_type
            