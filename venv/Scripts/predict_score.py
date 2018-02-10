import argparse
from face_feature_extractor import FaceFeatureExtractor
from scores_extractor import ScoresExtractor
from predictor import Predictor
import json
from collections import OrderedDict
import glob
import os

image_paths = ["C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\17_brad_pitt.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\20_Greg_Frers_0001.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\31_mila_kunis.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\63_Brooke_Adams_0001.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\76_Paula_Dobriansky_0001.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\98_Rachel_Griffiths_0001.jpg",
               "C:\\Users\Emi\PycharmProjects\FaceAnalyzer\dataset\\85_british_model_1.jpg"]

def predict_score(features, scores):
    predictor = Predictor(features, scores)
    tree = predictor.fuzzy_regression_tree()
    cart_tree = predictor.cart_tree()
    for image in image_paths:
        person = image.split("\\")[-1]
        score = scores_scaled.get(person)
        print("Analyzed person:",person)
        print("Average score (as rated by respondents: ",score)
        feature_extractor = FaceFeatureExtractor(image)
        img_features = feature_extractor.get_face_features()
        features_scaled = predictor.scale_features(img_features['features_values'])
        features_filtered = predictor.filter_scaled_pca_features(features_scaled)
        fuzzy_result = tree.classify([features_filtered])
        print("Clasic CART Tree score",cart_tree.classify(features_filtered))
        print("Fuzzy decision tree score:",fuzzy_result)
        print("==="*20)
    print("Analisys ended")


def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

def form_result(score1,score2):
    return (score1+score)/2


features = load_features_from_file('..\..\./features/features.json')
features_men = load_features_from_file('..\..\./features/features_men.json')
features_women = load_features_from_file('..\..\./features/features_women.json')

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('..\..\./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()

predict_score(features, scores_scaled)
