import math
import numpy as np
from copy import copy
import cv2
import string
import itertools
import imutils


class AlignerConfig:
    def __init__(self, args):
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
        self.old_engine_folder_name = args["engine_pair"][0]
        self.new_engine_folder_name = args["engine_pair"][1]


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
        and box["top"] < float(pos["top"]) + atol
        and box["top"] < float(pos["top"]) + atol
        and box["bottom"] > float(pos["bottom"]) - atol
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


def get_word_match_dict(template_tokens, image_tokens, match_words):
    """Find all words that occur in both the template and the image. Returns
    dictionaries mapping these words to their respective tokens in each doc."""
    image_matches = {}
    template_matches = {}

    for k in match_words:
        # Dictionary from key of word to all instances of word on the image
        image_matches[k] = [
            tok for tok in image_tokens if tok["text"].lower() == k.lower()
        ]

        # Dictionary from key of word to all instances of word on the template
        template_matches[k] = [
            tok for tok in template_tokens if tok["text"].lower() == k.lower()
        ]
    return image_matches, template_matches


def get_match_words_from_page(page_ocr, granularity="tokens"):
    # sourcery skip: de-morgan, inline-immediately-returned-variable
    tokens = [t["text"] for t in page_ocr[granularity]]
    filtered = [w for w in tokens if not w.lower() in string.punctuation]
    return filtered


def get_keypoints(
    image_matches, template_matches, image_size, template_size, debug=False
):
    """
    Given a set of words that occur in both the image and the candidate template,
    creates lists of cv2.DMatch and cv2.KeyPoint instances which can be used to
    construct a homography matrix
    Parameters:
        image_matches (dict): maps word occurring in both docs to list of
                              image tokens containing that word
        template_matches (dict): Same as above but with template
        image_size (dict): size in pixels with keys 'height', 'width'
        template_size (dict): size in pixels with keys 'height', 'width'
    Returns:
    kps: (KeyPoint list, KeyPoint list) of image and template keypoint lists
    matches: (DMatch list): list of created DMatch instances
    """
    kpsA = []
    kpsB = []
    matches = []

    # Sorting words because we later sort matches by distance, and that
    # order is not deterministic if there are multiple matches with
    # identical distances, which can happen if the difference between
    # top/bottom and left/right is identical for two matches
    for word in sorted(image_matches.keys()):
        ims = image_matches[word]
        temps = template_matches[word]

        # Product between every token in image and every token in template
        # that has the same word
        for im, temp in itertools.product(ims, temps):
            cv_match = cv2.DMatch(
                _distance=pixel_distance(im["nposition"], temp["nposition"]),
                _queryIdx=len(matches),
                _trainIdx=len(matches),
                _imgIdx=0,
            )

            matches.append(cv_match)

            # Keypoints on image
            kp_im = cv2.KeyPoint(
                x=image_size["width"]
                * (im["nposition"]["left"] + im["nposition"]["right"])
                / 2,
                y=image_size["height"]
                * (im["nposition"]["bottom"] + im["nposition"]["top"])
                / 2,
                size=1,
            )
            kpsA.append(kp_im)

            # Keypoints on template
            kp_temp = cv2.KeyPoint(
                x=template_size["width"]
                * (temp["nposition"]["left"] + temp["nposition"]["right"])
                / 2,
                y=template_size["height"]
                * (temp["nposition"]["bottom"] + temp["nposition"]["top"])
                / 2,
                size=1,
            )
            kpsB.append(kp_temp)

    return (kpsA, kpsB), matches


def keypoints(
    *,
    config,
    matches,
    kps,
    template_image=None,
    image=None,
    debug=False,
):
    """Analyze images with ORB and RANSAC.
    Parameters:
        config (dict): config parameters
        template_image: cv2 image object to be analyzed
        template: cv2 image object to be analyzed
        keepFraction (float) : how many matches to keep for alignment
        scale_factor (float) : scale image after alignment
        min_distance (float) : maximum distance (0-1 scale) to keep for alignment
        debug (bool) : if True display keypoints for manual check before alignment
        matches (cv2.DMatch list) : list of matches between keypoints to use
        kps (cv2.KeyPoint list) : list of keypoints to use
    Returns:
        Aligned image object.
    """
    kpsA, kpsB = kps

    # sort the matches by their distance (the smaller the distance, the
    # "more similar" the features are). Need to also sort by the queryIdx
    # so that the order is deterministic in case there are multiple
    # matches with the same distance
    matches = [m for m in matches if m.distance <= config.MAX_PIXEL_DISTANCE]
    matches = sorted(matches, key=lambda x: (x.distance, x.queryIdx))
    # keep only the top matches
    matches = matches[: config.NUM_KEYPOINTS_FOR_ALIGNMENT]

    if len(matches) < 5:
        return None, None

    # visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template_image, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1300)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x,y-coordinates) from the # top matches
    # we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for i, m in enumerate(matches):
        # indicate that the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points
    affine_method = (
        cv2.estimateAffinePartial2D if config.PARTIAL_AFFINE else cv2.estimateAffine2D
    )
    (H, mask) = affine_method(
        ptsA,
        ptsB,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransacReprojThreshold,
        maxIters=config.maxIters,
        refineIters=config.refineIters,
        confidence=config.confidence,
    )

    if debug:
        successful_matches = [
            match for mask_val, match in zip(mask, matches) if mask_val
        ]
        matchedVis = cv2.drawMatches(
            image, kpsA, template_image, kpsB, successful_matches, None
        )
        matchedVis = imutils.resize(matchedVis, width=1300)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    return H, mask


def extract_text_spans_from_box(
    config, template_width, template_height, image_tokens, box
):
    """
    Given a box on a template, extract image tokens that lie within the
    box.
    Parameters:
        template_width (int): prediction dict returned from model.predict()
        template_height (int): offset dict return from Aligner.acord_alignment_fix
        image_tokens (dict list): list of tokens on input image, as returned
            from OCR
        box (dict list): list of boxes on template - currently as returned
            from OmniPage template constructor
    """
    normalized_box = normalize_position(
        box["position"], template_width, template_height
    )
    # TODO: Fix this O(N^2) operation with something like a KD-Tree
    toks_in_box = list(
        filter(
            lambda tok: in_box(
                tok["nposition"],
                normalized_box,
                atol=config.TEXT_EXTRACTION_ATOL,
            ),
            image_tokens,
        )
    )
    toks = sorted(toks_in_box, key=lambda t: t["position"]["top"])
    text_spans, text = order_text_spans_from_zone(config, box, toks)

    return {
        "label": box["label"],
        "text_spans": text_spans,
        "text": text,
        "page_num": box["page_num"],
    }, toks


def order_text_spans_from_zone(config, box, toks):
    """
    Given the extracted tokens from a box, orders tokens by position and
    returns the GlobalTokenSpan information (start, end, page_num)
    for each piece of text, as well as the joined text string
    """
    if len(toks) == 0:
        return [], ""
    line_breaks = [0] + [
        i
        for i in range(1, len(toks))
        if toks[i]["position"]["top"] - toks[i - 1]["position"]["top"]
        > config.NEWLINE_PIXELS_THRESHOLD
    ]

    # split tokens into separate lines
    ordered_tokens = []
    if len(line_breaks) == 1:
        ordered_tokens.append(sorted(toks, key=lambda t: t["position"]["left"]))
    else:
        for idx1, idx2 in zip(line_breaks, line_breaks[1:]):
            line = toks[idx1:idx2]
            ordered_tokens.append(sorted(line, key=lambda t: t["position"]["left"]))
        last_line = toks[line_breaks[-1] :]
        ordered_tokens.append(sorted(last_line, key=lambda t: t["position"]["left"]))
    text = "\n".join(" ".join(tok["text"] for tok in line) for line in ordered_tokens)

    # Return start/end indices for each group of tokens. If tokens are
    # contiguous, join them
    text_spans = []
    for line in ordered_tokens:
        prev_end = None
        for token in line:
            if prev_end is not None and token["page_offset"]["start"] - 1 == prev_end:
                # Join with previous token rather than creating new text span
                text_spans[-1]["end"] = token["page_offset"]["end"]
            else:
                text_spans.append(
                    {
                        "start": token["page_offset"]["start"],
                        "end": token["page_offset"]["end"],
                        "page_num": box["page_num"],
                    }
                )
            prev_end = token["page_offset"]["end"]

    return text_spans, text
