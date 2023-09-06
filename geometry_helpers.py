import math
from copy import copy

import cv2
import numpy as np


class AlignerConfig:
    def __init__(self, args):
        self.debug = args['debug']
        self.min_keypoint_match_ratio = args['min_keypoint_match_ratio']
        self.MAX_PIXEL_DISTANCE = args["MAX_PIXEL_DISTANCE"]
        self.NUM_KEYPOINTS_FOR_ALIGNMENT = args["NUM_KEYPOINTS_FOR_ALIGNMENT"]
        self.PARTIAL_AFFINE = args["PARTIAL_AFFINE"]
        self.maxIters = args["maxIters"]
        self.refineIters = args["refineIters"]
        self.confidence = args["confidence"]
        self.ransacReprojThreshold = args["ransacReprojThreshold"]
        self.TEXT_EXTRACTION_ATOL = args["TEXT_EXTRACTION_ATOL"]
        self.NEWLINE_PIXELS_THRESHOLD = args["NEWLINE_PIXELS_THRESHOLD"]
        self.granularity = args["granularity"]
        self.old_engine_folder_name = args["source_folder"]
        self.new_engine_folder_name = args["target_folder"]


def pixel_distance(pos1, pos2):
    x1 = (pos1["left"] + pos1["right"]) / 2
    y1 = (pos1["top"] + pos1["bottom"]) / 2

    x2 = (pos2["left"] + pos2["right"]) / 2
    y2 = (pos2["top"] + pos2["bottom"]) / 2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_position(position, page_width, page_height):
    return {
        "top": position["top"] / page_height,
        "bottom": position["bottom"] / page_height,
        "left": position["left"] / page_width,
        "right": position["right"] / page_width,
    }


def in_box(pos, box, atol=0):
    return (
        box["left"] < float(pos["left"]) + atol 
        and box["right"] > float(pos["right"]) - atol
        and box["top"] < float(pos["top"]) + atol * 0.5
        and box["bottom"] > float(pos["bottom"]) - atol * 0.5
    )


def normalize_tokens(tokens, page_size):
    for token in tokens:
        token["nposition"] = normalize_position(
            token["position"], page_size["width"], page_size["height"]
        )
        token["text_lower"] = token["text"].lower()
    return tokens


def apply_transform(left, top, right, bot, H):
    bbox = np.array(
        [[[left, bot]], [[right, bot]], [[left, top]], [[right, top]]], dtype=np.float32
    )
    transformed = np.ceil(cv2.perspectiveTransform(bbox, H))
    left = int(min(transformed[0, 0][0], transformed[2, 0][0]))
    bot = int(max(transformed[0, 0][1], transformed[1, 0][1]))
    right = int(max(transformed[1, 0][0], transformed[3, 0][0]))
    top = int(min(transformed[2, 0][1], transformed[3, 0][1]))
    return left, top, right, bot


def transform_tokens(H, token_list):
    transformed_tokens = []
    for t in token_list:
        transformed = copy(t)
        bbLeft_t, bbTop_t, bbRight_t, bbBot_t = apply_transform(
            t["position"]["bbLeft"],
            t["position"]["bbTop"],
            t["position"]["bbRight"],
            t["position"]["bbBot"],
            H,
        )
        transformed["position"]["bbLeft"] = bbLeft_t
        transformed["position"]["bbTop"] = bbTop_t
        transformed["position"]["bbRight"] = bbRight_t
        transformed["position"]["bbBot"] = bbBot_t

        left, top, right, bot = apply_transform(
            t["position"]["left"],
            t["position"]["top"],
            t["position"]["right"],
            t["position"]["bottom"],
            H,
        )
        transformed["position"]["left"] = left
        transformed["position"]["top"] = top
        transformed["position"]["right"] = right
        transformed["position"]["bottom"] = bot
        transformed_tokens.append(transformed)
    return transformed_tokens


def get_word_match_dict(old_tokens, new_tokens, match_words):
    """Find all words that occur in both the template and the image. Returns
    dictionaries mapping these words to their respective tokens in each doc."""
    old_matches = {}
    new_matches = {}

    for k in match_words:
        old_matches[k] = [
            tok for tok in old_tokens if tok["text"].lower() == k.lower()
        ]

        new_matches[k] = [
            tok for tok in new_tokens if tok["text"].lower() == k.lower()
        ]
    return old_matches, new_matches

