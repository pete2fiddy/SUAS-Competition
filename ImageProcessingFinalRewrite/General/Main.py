from Stat.EigenImageProjector import EigenImageProjector
from FileLoading.DataLoader import DataLoader
from PIL import Image
import NoiseReduction.GaussianBlur as GaussianBlur
from EdgeProcessing.SobelEdge import SobelEdge
import EdgeProcessing.CannyEdge as CannyEdge
from ImgStat.KMeans import KMeans
import Color.ColorMath as ColorMath
from Color.ColorSplitter import ColorSplitter
from General.Target import Target
from DataMine.KNearestNeighbors import KNearestNeighbors
from FileLoading.Categorizer import Categorizer
from Character.SWT import SWT
from PIL import ImageOps
import timeit
from DataMine.ZScore import ZScore
import EdgeProcessing.HarrisCorner as HarrisCorner
import Color.TargetColorReader as TargetColorReader

'''
kmeans_test_img = Image.open("/Users/phusisian/Desktop/Senior year/SUAS/Object images test/L Rectangle Cropped.png")
kmeans_test_img = kmeans_test_img.resize((kmeans_test_img.size[0]/2, kmeans_test_img.size[1]/2))
test_kmeans = KMeans.init_with_img(kmeans_test_img, kmeans_test_img.load(), 3, 20, 3)
ColorMath.get_img_rounded_to_colors(kmeans_test_img, kmeans_test_img.load(), test_kmeans.get_cluster_origins_int()).show()
'''


shape_img = Image.open("/Users/phusisian/Desktop/Senior year/SUAS/Object images test/Small Crops/N Triangle Cropped.jpg")#Object images test/Small Crops/N Triangle Cropped.jpg")#400 crop triangle 6480x4320.jpg")
scale = 1#1.0/2.0
shape_img = shape_img.resize((int(shape_img.size[0]*scale), int(shape_img.size[1]*scale)))
target = Target(shape_img, shape_img.load())

print("Target: \n" + str(target))













'''
letter_img = Image.open("/Users/phusisian/Desktop/Senior year/SUAS/Object images/LetterR.png").convert('L')
letter_img = letter_img.resize((letter_img.size[0]/3, letter_img.size[1]/3))
letter_img.show()
letter_sobel = SobelEdge(letter_img)
letter_sobel.get_gradient_mag_img().show()
letter_canny_img = CannyEdge.get_canny_img(letter_sobel, (10, 20))
letter_canny_img.show()
letter_canny_image = letter_canny_img.load()
letter_SWT = SWT(letter_sobel, letter_canny_img, letter_canny_image)
'''


'''

letter_img = Image.open("/Users/phusisian/Desktop/Senior year/SUAS/PCATesting/LetterImgs/English/Fnt/Sample016/img016-00562.png").convert('L')
letter_img = letter_img.resize((40, 40), Image.NEAREST)
letter_img = ImageOps.invert(letter_img)
#letter_img = GaussianBlur.get_gaussian_filtered_bw_img(letter_img, letter_img.load(), 3, 2)
letter_img.show()
letter_categorizer = Categorizer("/Users/phusisian/Desktop/Senior year/SUAS/PCATesting/SUASLetterImgs/DataNormalized", 50, ZScore)
start_time = timeit.default_timer()
print("sorted list z score: " + str(letter_categorizer.get_algorithm_return_smallest_to_large(letter_img, None)))
print("time elapsed: " + str(timeit.default_timer() - start_time))'''

'''
plaid_img = Image.open("/Users/phusisian/Desktop/Senior year/SUAS/Object images/startest.jpg").convert('L')
plaid_img = GaussianBlur.get_gaussian_filtered_bw_img(plaid_img, plaid_img.load(), 3, 1)
plaid_img.show()
plaid_sobel = SobelEdge(plaid_img)
plaid_corners = HarrisCorner.get_corners_over_threshold(plaid_sobel, 3, 1, 1000000)
HarrisCorner.draw_corners(plaid_corners, plaid_img.load(), 255)
plaid_img.show()
print("plaid corners: " + str(plaid_corners))'''




