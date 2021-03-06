from SUASImageParser.ADLC import ADLCParser
from SUASImageParser.utils.color import bcolors
from SUASImageParser.optimizers.new_optimizer import OptimizerServer
import cv2
import numpy as np
import os
import timeit
import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

class ADLCOptimizer:

    def __init__(self, **kwargs):
        self.debug = kwargs.get("debug", False)

    def optimize(self, output_log_file, img_directory, multithread):
        """
        This creates the optimization data and optimizes the ADLC parser to
        perform with the best results. It then saves that data so that future
        runs of the program use the optimized settings.
        """
        # get starting parameters
        # To set the parameters, provide the following
        # "PARAM_NAME" : {
        #   "STARTING_VAL" : STARTING_VALUE,
        #   "MIN_VAL" : MINIMUM_VALUE,
        #   "MAX_VAL" : MAXIMUM_VALUE,
        #   "ACCURACY" : ACCURACY
        # }
        # NOTE: ACCURACY is the number of decimal places you would like the
        #   optimized parameter to be hypertuned to
        parameters = {
            "THRESH_VALUE" : {
                "STARTING_VAL" : 0.0,
                "MIN_VAL" : 0.0,
                "MAX_VAL" : 255.0,
                "ACCURACY" : 0
            },
            "SCALE_FACTOR" : {
                "STARTING_VAL" : 0.0,
                "MIN_VAL" : 0.0,
                "MAX_VAL" : 1.0,
                "ACCURACY" : 2
            }
        }

        self.output_log_file = output_log_file
        self.scenario_log = {
            "scenarios" : [],
            "scenario_index" : 0,
            "total_time" : 0.0,
            "best" : {
                "best_params" : {},
                "best_score" : -1
            }
        }

        if os.path.exists(self.output_log_file):
            with open(self.output_log_file, 'r') as data_file:
                self.scenario_log = json.load(data_file)

        self.img_directory = img_directory

        # return optimized parameters
        optimized_params, score = self.run_optimization(parameters)
        return optimized_params

    def run_optimization(self, parameters):
        """
        Run optimization on a given set of parameters and images.
        """
        best_score = self.scenario_log["best"]["best_score"]
        best_params = self.scenario_log["best"]["best_params"]
        scenarios = self.create_scenarios(0, [], parameters)

        if self.debug:
            print(bcolors.INFO + "[Info]" + bcolors.ENDC + " Calculated " + str(len(scenarios)) + " scenarios to run")

        self.scenario_index = self.scenario_log["scenario_index"]

        if self.scenario_index != 0 and self.debug:
            print(bcolors.INFO + "[Info]" + bcolors.ENDC + " Resuming optimization from log file")

        self.optimization_server = OptimizerServer(debug=True)
        completed_scenarios = self.optimization_server.serve(scenarios, self.img_directory)
        for completed_scenario in completed_scenarios:
            # Each completed scenario takes the form of:
            # [ scenario_index, score, scenario ]
            self.scenario_log["scenarios"].append(completed_scenario)

            if completed_scenario[1] > best_score:
                print("FOUND A BETTER SCORE")
                best_params = completed_scenario[2]
                best_score = completed_scenario[1]
                self.scenario_log["best"]["best_params"] = best_params
                self.scenario_log["best"]["best_score"] = best_score

        return best_params, best_score

    def create_scenarios(self, index, scenarios, parameters):
        """
        Create scenarios for optimization
        """
        for_index = 0
        for parameter in parameters:
            if for_index == len(parameters) - 1:
                for param_index_val in np.arange(parameters[parameter]["MIN_VAL"], parameters[parameter]["MAX_VAL"], 10**(-1.0 * parameters[parameter]["ACCURACY"])):
                    parameters[parameter]["STARTING_VAL"] = param_index_val

                    scenarios.append(self.extrapolate_scenario(parameters))
            elif for_index == index:
                for param_index_val in np.arange(parameters[parameter]["MIN_VAL"], parameters[parameter]["MAX_VAL"], 10**(-1.0 * parameters[parameter]["ACCURACY"])):
                    parameters[parameter]["STARTING_VAL"] = param_index_val

                    scenarios = self.create_scenarios(for_index + 1, scenarios, parameters)

            for_index += 1

        return scenarios

    def extrapolate_scenario(self, parameters):
        """
        Extrapolate a scenario from a set of parameters.
        """
        scenario = {}
        for parameter in parameters:
            scenario[parameter] = parameters[parameter]["STARTING_VAL"]

        return scenario

    def load_images(self, img_directory=None):
        """
        Load an image and its known solutions
        """
        if img_directory == None:
            raise ValueError('load_image() cannot be passed a directory with type None')

        if self.debug:
            num_images = 0
            for dir in next(os.walk(img_directory))[1]:
                for file in next(os.walk(os.path.join(img_directory, dir)))[2]:
                    if file.endswith(".jpg"):
                        num_images += 1
            print(bcolors.INFO + "[Info]" + bcolors.ENDC + " Loading " + str(num_images) + " images")

        # The storage for images uses the following structure:
        # [
        #   [full_image, target_1, target_2, target_3, target_4],
        #   [full_image, target_1, target_2, target_3, target_4]
        # ]
        images = []
        for dir in next(os.walk(img_directory))[1]:
            new_img_set = []
            for file in next(os.walk(os.path.join(img_directory, dir)))[2]:
                path = os.path.join(img_directory, dir, file)

                if "image" in file and file.endswith(".jpg"):
                    new_img_set = [cv2.imread(path)] + new_img_set
                elif file.endswith(".txt"):
                    with open(path) as data_file:
                        new_img_set.append(json.load(data_file))

            images.append(new_img_set)

        if self.debug:
            print(bcolors.INFO + "[Info]" + bcolors.ENDC + " Images successfully loaded")

        return images

    def run_params(self, images, parameters):
        """
        Run an image through the ADLC Parser, score all of the components found,
        return the resulting information.
        """
        ADLC_parser = ADLCParser()
        if self.debug:
            print("\n" + bcolors.INFO + "[Info]" + bcolors.ENDC + " Running a new scenario...")

        scores = 0.0
        img_index = 0
        for image in images:
            img_index += 1
            test_image = image[0]
            ADLC_parser.setup(parameters)

            if self.debug:
                start_time = timeit.default_timer()

            targets, _, contours = ADLC_parser.parse(test_image)
            score = self.score(contours, image[1:])
            scores += score

            if score > 0.0 and debug:
                for target in targets:
                    cv2.imshow("target", target)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if self.debug:
                end_time = timeit.default_timer()
                image_run_time = end_time - start_time
                print(bcolors.INFO + "[Info]" + bcolors.ENDC + " Image number " + str(img_index) + " scored a " + str(score*100) + "% (" + str(image_run_time) + " seconds)")

        scores = scores / float(len(images))
        if self.debug:
            print(bcolors.INFO + "[Info]" + bcolors.ENDC + " The scenario got a score of " + str(scores*100) + "%")

        return scores

    def score(self, test, correct):
        """
        Compare proposed target and the actual solution.

        Returns:
        - How many of the targets are correctly found. For every FA
            (False Alarm)
        """
        score = 0.0
        for test_img in test:
            num_pixels = float(len(test_img))
            best_img_score = 0.0
            x,y,w,h = cv2.boundingRect(test_img)

            for correct_img in correct:
                img_score = 0.0
                area = float(correct_img["x_finish"] - correct_img["x_start"]) * float(correct_img["y_finish"] - correct_img["y_start"])

                SI = max(0, max(x + w, correct_img["x_finish"]) - min(x, correct_img["x_start"])) * max(0, max(y + h, correct_img["y_finish"]) - min(y, correct_img["y_start"]))
                img_score = float(h * w) + area - SI

                if img_score > best_img_score:
                    best_img_score = img_score

            score += best_img_score

        return score / float(len(correct))
