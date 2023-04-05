import numpy as np
import os
import ast
import imutils
import cv2
import pandas as pd
import json
import pandas as pd
import json
import string
import itertools
from indico_toolkit.association import ExtractedTokens
from copy import copy
import logging
import yaml
from fire import Fire
from indico.client import GraphQLRequest
import tqdm

class GraphQLMagic(GraphQLRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(query=self.query, variables=kwargs)


logging.basicConfig(
    filename="ocr_migration.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOGGING_LEVEL", "INFO")
)

from geometry_helpers import (
    AlignerConfig,
    normalize_position,
    normalize_tokens,
    in_box,
    get_word_match_dict,
    transform_tokens,
    pixel_distance,
)

from comparison_helpers import summarize_results, convert_to_excel


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
    }


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
            if prev_end is not None and token["doc_offset"]["start"] - 1 == prev_end:
                # Join with previous token rather than creating new text span
                text_spans[-1]["end"] = token["doc_offset"]["end"]
            else:
                text_spans.append(
                    {
                        "start": token["doc_offset"]["start"],
                        "end": token["doc_offset"]["end"],
                        "page_num": box["page_num"],
                    }
                )
            prev_end = token["doc_offset"]["end"]

    return text_spans, text


def run_all_pages_for_doc(
    file,
    new_ocr_for_file,
    old_ocr_for_file,
    aligner_config,
    debug=False,
):
    new_images = json.loads(new_ocr_for_file["image_files"].values[0])
    new_ocr = json.loads(new_ocr_for_file["ocr"].values[0])

    old_ocr = json.loads(old_ocr_for_file["ocr"].values[0])
    old_images = json.loads(old_ocr_for_file["image_files"].values[0])
    labels = json.loads(old_ocr_for_file["labels"].values[0])

    label_to_token_by_page = {}
    for p_0, n_0 in zip(old_ocr, new_ocr):
        old_page_ocr = p_0["pages"][0]
        page_number = old_page_ocr["page_num"]
        min_offset = old_page_ocr["doc_offset"]["start"]
        max_offset = old_page_ocr["doc_offset"]["end"]

        old_size = old_page_ocr["size"]
        old_tokens = p_0["tokens"]
        norm_old_tokens = normalize_tokens(old_tokens, old_size)

        new_page_ocr = n_0["pages"][0]
        new_size = new_page_ocr["size"]
        new_tokens = n_0["tokens"]
        norm_new_tokens = normalize_tokens(new_tokens, new_size)

        if aligner_config.granularity == "tokens":
            norm_old_points = norm_old_tokens
            norm_new_points = norm_new_tokens

        else:
            norm_old_points = normalize_tokens(p_0["chars"], old_size)
            norm_new_points = normalize_tokens(n_0["chars"], new_size)

        matches = get_match_words_from_page(p_0, aligner_config.granularity)
        new_match, old_match = get_word_match_dict(
            norm_new_points, norm_old_points, matches
        )

        kps, cv_matches = get_keypoints(old_match, new_match, old_size, new_size)

        original_image = cv2.imread(old_images[page_number])
        new_image = cv2.imread(new_images[page_number])
        homography, mask = keypoints(
            config=aligner_config,
            kps=kps,
            matches=cv_matches,
            template_image=new_image,
            image=original_image,
            debug=debug,
        )
        if homography is None:
            message = f"RANSAC failed for {file} on page {page_number}, using identity matrix."
            print(message)
            logging.warning(message)
            affine_warp = np.identity(3)
        else:
            affine_warp = np.vstack([homography, [0, 0, 1]])

        ex_tokens, transformed = transform_and_extract(
            labels, min_offset, max_offset, old_tokens, affine_warp
        )

        label_token_map = []
        for t, o in zip(transformed, ex_tokens.extracted_tokens):
            extracted_spans_for_label = extract_text_spans_from_box(
                aligner_config,
                new_size["width"],
                new_size["height"],
                norm_new_tokens,
                t,
            )
            label_token_map.append(
                {"token": t, "spans": extracted_spans_for_label, "original_token": o}
            )

        label_to_token_by_page[page_number] = label_token_map

    return label_to_token_by_page


def transform_and_extract(labels, min_offset, max_offset, old_tokens, affine_warp):
    labels_on_page = [
        l for l in labels if l["end"] <= max_offset and l["start"] >= min_offset
    ]
    ex_tokens = ExtractedTokens(labels_on_page)
    ex_tokens.collect_tokens(old_tokens, raise_for_no_match=False)
    transformed = transform_tokens(affine_warp, ex_tokens.extracted_tokens)
    return ex_tokens, transformed


def run_all_docs(config, k=None):
    logging.info("Loading new dataset...")
    new_directory = config.new_engine_folder_name
    new_ocr_path = f"{new_directory}/all_labels.csv"
    new_ocr_df = pd.read_csv(new_ocr_path)
    new_ocr_df["file_name"] = new_ocr_df['document_path'].apply(lambda x: os.path.basename(x))
    new_file_names = set(new_ocr_df["file_name"])

    logging.info("Loading old dataset...")
    old_directory = config.old_engine_folder_name
    old_ocr_path = f"{old_directory}/all_labels.csv"
    old_ocr_df = pd.read_csv(old_ocr_path)
    old_ocr_df["file_name"] = old_ocr_df['document_path'].apply(lambda x: os.path.basename(x))
    old_ocr_df = old_ocr_df[old_ocr_df["labels"].notna()]
    if not len(old_ocr_df):
        raise ValueError("No files have valid labels")
    old_file_names = set(old_ocr_df["file_name"])

    common_file_names = old_file_names & new_file_names
    if len(old_file_names) > len(common_file_names):
        print(
            f"The following files are present in the old ocr, but not in the new: {','.join(list(old_file_names - common_file_names))}."
        )

    mappings_by_file_name = {}
    for f in tqdm.tqdm(list(common_file_names)[:k]):
        new_ocr_for_file = new_ocr_df[new_ocr_df["file_name"] == f]
        old_ocr_for_file = old_ocr_df[old_ocr_df["file_name"] == f]
        matched_labels_for_doc = run_all_pages_for_doc(
            f,
            new_ocr_for_file,
            old_ocr_for_file,
            config,
            debug=False,
        )
        mappings_by_file_name[f] = matched_labels_for_doc

    return mappings_by_file_name


def run(config, new_dataset_id, num_docs=None, api_key_path="prod_api_token.txt", indico_host="app.indico.io", summary_file="./summary.xlsx"):
    with open(config, "r") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    aligner_config = AlignerConfig(config_dict)
    all_results = run_all_docs(aligner_config, k=num_docs)
    migrated_labels_csv = os.path.join(aligner_config.new_engine_folder_name, 'revised_labels.json')
    with open(migrated_labels_csv, 'w') as fd:
        json.dump(all_results, fd)



if __name__ == "__main__":
    Fire(run)
