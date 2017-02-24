from ImgStat.KMeans import KMeans
from Color.ColorSplitter import ColorSplitter
import NoiseReduction.GaussianBlur as GaussianBlur
from EdgeProcessing.SobelEdge import SobelEdge
import EdgeProcessing.CannyEdge as CannyEdge
from Geometry.PolarSideCounter import PolarSideCounter
import Color.TargetColorReader as TargetColorReader
from ImgStat.SimplePCA import SimplePCA
from TargetTrait.TargetDirection import TargetDirection
from PIL import ImageDraw
from TargetTrait.ShapeType import ShapeType
from math import degrees
from PIL import Image
from FileLoading.Categorizer import Categorizer
from DataMine.ZScore import ZScore
from Color.ColorLayer import ColorLayer
import ImageOperation.Crop as Crop
import ImageOperation.Scale as Scale
import ImageOperation.Paste as Paste

class Target:
    BLUR_SHAPE_KERNELSIZE = 3
    BLUR_SHAPE_STD_DEV = 2
    CANNY_SHAPE_THRESHOLDS = (20, 40)
    CANNY_TOTAL_THRESHOLDS = (20, 40)
    BLUR_TOTAL_KERNELSIZE = 5
    BLUR_TOTAL_STD_DEV = 5
    PCA_LETTER_DIM = (40,40)
    LETTER_RESIZE_HEIGHT = int(PCA_LETTER_DIM[1]*0.81666666666667)
    KMEANS_RUN_TIMES = 20
    KMEANS_STEP = 3
    
    def __init__(self, img_in, image_in):
        self.target_img = img_in
        self.target_image = image_in
        self.init_edge_imgs()
        self.init_color_splitter_and_layers()
        
        '''to improve, remove the background color from the image then do canny on only the shape and letter.
        Will eliminate noise from background. Worth noting that the noise from background probably matters very
        little in the calculation'''
        self.pca = SimplePCA.init_with_canny_img(self.target_canny_img, self.target_canny_img.load())
        
        '''initializing target traits'''
        self.init_target_colors()
        self.init_shape_type()
        self.init_target_direction()
        self.init_letter_recognition_imgs()
        self.run_possible_imgs_through_letter_recognition()
      
    def init_edge_imgs(self):
        bw_target_img = self.target_img.copy().convert('L')
        gaussian_img = GaussianBlur.get_gaussian_filtered_bw_img(bw_target_img, bw_target_img.load(), Target.BLUR_TOTAL_KERNELSIZE, Target.BLUR_TOTAL_STD_DEV)
        target_sobel_edge = SobelEdge(gaussian_img)
        self.target_canny_img = CannyEdge.get_canny_img(target_sobel_edge, Target.CANNY_TOTAL_THRESHOLDS)
    
    def init_color_splitter_and_layers(self):
        self.color_splitter = ColorSplitter.init_with_kmeans(self.target_img, self.target_image, 3, Target.KMEANS_RUN_TIMES, Target.KMEANS_STEP)
        self.color_splitter.sort_by_area()
        self.unfilled_color_layers = self.color_splitter.get_color_layers().clone()
        self.color_splitter.sort_by_area_then_fill_gaps()
        self.color_layers = self.color_splitter.get_color_layers()
        self.shape_layer = self.get_shape_color_layer()
        self.init_letter_layer()
        
    def init_letter_layer(self):
        self.letter_layer = self.color_layers[0]
        self.remove_possible_false_border_around_letter()
    
    def remove_possible_false_border_around_letter(self):
        '''by appending the letter layer, which may have an incorrect border around it as a result of KMeans,
        onto the background layer, the false border will connect with the background layer. As a result, 
        flood-filling any place that is the background and not the letter will remove the entire background
        from the appended letter-to-background image'''
        unfilled_background_layer = self.unfilled_color_layers[len(self.unfilled_color_layers) - 1]
        background_and_letter_layer = unfilled_background_layer.get_layer_filled_with_layer(self.letter_layer)
        '''chose (0,0) as a start because that seemed guaranteed to have the background in it. However,
        I can see times where this is not the case, and it may be necessary to add a method to ColorLayer
        that will return a pixel (doesn't really matter which) that is a part of the color layer, and 
        the flood-filling will begin there'''
        
        ImageDraw.floodfill(background_and_letter_layer.get_layer_img(), (0,0), 0)
        self.letter_layer = background_and_letter_layer
    
    def init_target_colors(self):
        self.TARGET_SHAPE_COLOR = TargetColorReader.get_closest_target_color(self.shape_layer.get_color())
        self.TARGET_CHARACTER_COLOR = TargetColorReader.get_closest_target_color(self.letter_layer.get_color())

    def init_shape_type(self):
        shape_img = self.shape_layer.get_layer_img().copy()
        self.shape_type = ShapeType(shape_img, self.pca)
        
        self.TARGET_SHAPE = self.shape_type.get_shape_type()
    
    '''assuming the color splitter is sorted by top to bottom layer, the layer with the shape should always be
    2nd from the end, because it is directly after the background'''
    def get_shape_color_layer(self):
        return self.color_layers[len(self.color_layers)-2]
    
    def init_target_direction(self):
        self.target_direction = TargetDirection(self.pca.get_eigenvectors(), self.shape_type.get_polar_side_counter())


    def init_letter_recognition_imgs(self):
        self.possible_imgs = self.target_direction.get_letter_imgs_rotated_to_possible_directions(self.letter_layer.get_layer_img())
        for i in range(0, len(self.possible_imgs)):
            self.possible_imgs[i] = (self.get_letter_img_resized_to_PCA_dims(self.possible_imgs[i][0]), self.possible_imgs[i][1], self.possible_imgs[i][2])

    def get_letter_img_resized_to_PCA_dims(self, letter_img):
        resized_letter_img = Crop.get_bw_img_cropped_to_bounds(letter_img, letter_img.load())
        resized_letter_img.show()
        resized_letter_img = Scale.scale_img_to_height(resized_letter_img, Target.LETTER_RESIZE_HEIGHT)
        base_img = Image.new('L', Target.PCA_LETTER_DIM, 0)
        offset = ((Target.PCA_LETTER_DIM[0]/2) - (resized_letter_img.size[0]/2), (Target.PCA_LETTER_DIM[1]/2) - (resized_letter_img.size[1]/2))
        Paste.paste_img_onto_img(resized_letter_img, base_img, offset)
        resized_letter_img = base_img
        return resized_letter_img

    def run_possible_imgs_through_letter_recognition(self):
        letter_categorizer = Categorizer("/Users/phusisian/Desktop/Senior year/SUAS/PCATesting/SUASLetterImgs/DataNormalized", 50, ZScore)
        
        best_fit_character = ""
        best_fit_score = 10000
        best_fit_direction = ""
        for i in range(0, len(self.possible_imgs)):
            iter_scores = letter_categorizer.get_algorithm_return_smallest_to_large(self.possible_imgs[i][0], None)
            iter_letter = iter_scores[0][0]
            iter_score = iter_scores[0][1]
            if iter_score < best_fit_score:
                best_fit_score = iter_score
                best_fit_character = iter_letter
                best_fit_direction = self.possible_imgs[i][1]
        
        self.TARGET_COMPASS_ORIENTATION = best_fit_direction
        self.TARGET_CHARACTER = best_fit_character
    
    def __repr__(self):
        out = ("orientation: " + str(self.TARGET_COMPASS_ORIENTATION) + "\n"
               + "shape: " + str(self.TARGET_SHAPE) + "\n"
               + "background_color: " + str(self.TARGET_SHAPE_COLOR) + "\n"
               + "alphanumeric: " + str(self.TARGET_CHARACTER) + "\n"
               + "alphanumeric_color: " + str(self.TARGET_CHARACTER_COLOR) + "\n")
        return out
        