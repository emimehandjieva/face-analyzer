import glob
import os
import json
from face_feature_extractor import FaceFeatureExtractor

features = {}
features_men = {}
features_women = {}

# gets name of all image files
all_images = glob.glob(os.path.realpath('..\..\./dataset/*.jpg'))
# same for women
all_women = glob.glob(os.path.realpath('..\..\./dataset/women/*.jpg'))

# same for men
all_men = glob.glob(os.path.realpath('..\..\./dataset/men/*.jpg'))


def extract_and_load_to_file(features, features_men, features_women):
    for filename in all_images:
        print(filename)
        # feed image to class that can analyze it
        next_feature_extractor = FaceFeatureExtractor(filename)

        # get all the relevant info for analysis and save it as an element in the features array
        features[os.path.basename(filename)] = next_feature_extractor.get_face_features()

    # same for women
    for filename in all_women:
        features_women[os.path.basename(filename)] = features[os.path.basename(filename)]

    # same for men
    for filename in all_men:
        features_men[os.path.basename(filename)] = features[os.path.basename(filename)]

    with open('features_women.json', 'w') as outfile:
        json.dump(features_women, outfile)

    with open('features_men.json', 'w') as outfile:
        json.dump(features_men, outfile)

    with open('features.json', 'w') as outfile:
        json.dump(features, outfile)


extract_and_load_to_file(features, features_men, features_women)