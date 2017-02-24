import ImgVector.VectorMath as VectorMath
import Array.ArrayHelper as ArrayHelper
import numpy
from math import sin, cos, atan2,pi, degrees

class TargetDirection:
    def __init__(self, eigenvectors_in, polar_side_counter_in):
        self.eigenvectors = eigenvectors_in
        self.polar_side_counter = polar_side_counter_in
        self.maximums = self.polar_side_counter.get_maximums()
        self.num_maxes = len(self.polar_side_counter.get_maximums())
        self.init_direction()
        
    def init_direction(self):
        possible_values = (self.eigenvectors[0], self.eigenvectors[0]*-1)
        '''if side max radius corner number is odd, shape is oriented so that one of its corner subtends
         the angle it is rotated. You can combine with PCA axis to find the correct corner this applies. 
         This should be direction angle'''
        if self.num_maxes%2 == 1:
            self.direction = AngleRounder.round_to_compass_angle(self.get_max_closest_to_possible_values(possible_values).get_angle())
            self.num_directions = 1
        else:
            self.direction = [AngleRounder.round_to_compass_angle(VectorMath.get_angle_of_2d_vector(possible_values[0])),
                              AngleRounder.round_to_compass_angle(VectorMath.get_angle_of_2d_vector(possible_values[1]))]
            self.num_directions = 2
            
        print(self.direction)
    
    def get_letter_imgs_rotated_to_possible_directions(self, letter_img):
        if self.num_directions == 1:
            return [(letter_img.rotate(degrees(self.direction[1]), expand=True), self.direction[0], self.direction[1])]
        else:
            angle1 = self.direction[0][1]
            angle2 = self.direction[1][1]
            img1 = (letter_img.rotate(degrees(angle1), expand=True), self.direction[0][0], self.direction[0][1])
            img2 = (letter_img.rotate(degrees(angle2), expand=True), self.direction[1][0], self.direction[1][1])
            return [img1, img2]
            
    
    def get_direction(self):
        return self.direction
    
    def get_angle(self):
        return self.direction[:][1]
    
    def get_max_closest_to_possible_values(self, possible_values):
        angle_differences = []
        '''a really big value'''
        smallest_difference = 100
        smallest_index = 0
        for i in range(0, len(possible_values)):
            sorted_differences = sorted(self.maximums, key=lambda radius_angle : abs(VectorMath.get_angle_between_vectors(radius_angle.get_unit_vector(), possible_values[i])))
            difference = abs(VectorMath.get_angle_between_vectors(sorted_differences[0].get_unit_vector(), possible_values[i]))
            if difference < smallest_difference:
                smallest_difference = difference
                smallest_index = i
            angle_differences.append(sorted_differences[0])
        return angle_differences[smallest_index]
    
    
class AngleRounder:
    COMPASS_ANGLES = [("E", 0), ("NE", pi/4.0), ("N", pi/2.0), ("NW", 3.0*pi/4.0), ("W", pi), ("SW", 5.0*pi/4.0), ("S", 3.0*pi/2.0), ("SE", 7.0*pi/4.0)]
    @staticmethod
    def round_to_compass_angle(angle):
        sorted_angles = sorted(AngleRounder.COMPASS_ANGLES, key = lambda compass_angle : abs(compass_angle[1]-angle)%(2.0*pi))
        return sorted_angles[0]
    
    