import itertools
import json
import logging
import os
import string
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import yaml
from fire import Fire
from indico_toolkit.association import ExtractedTokens

from comparison_helpers import convert_to_excel, summarize_results
from geometry_helpers import (
    AlignerConfig,
    get_word_match_dict,
    in_box,
    normalize_tokens,
    pixel_distance,
    transform_tokens,
)

logging.basicConfig(
    filename="ocr_migration.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOGGING_LEVEL", "INFO"),
)


def get_match_words_from_page(page_ocr, granularity="tokens"):
    tokens = [t["text"] for t in page_ocr[granularity]]
    filtered = [w for w in tokens if not w.lower() in string.punctuation and len(w.strip()) > 3]
    return filtered


def get_keypoints(
    old_matches, new_matches, old_size, new_size
):
    """
    Given a set of words that occur in both the image and the candidate template,
    creates lists of cv2.DMatch and cv2.KeyPoint instances which can be used to
    construct a homography matrix
    Parameters:
        old_matches (dict): maps word occurring in both docs to list of
                              image tokens containing that word
        new_matches (dict): Same as above but with template
        old_size (dict): size in pixels with keys 'height', 'width'
        new_size (dict): size in pixels with keys 'height', 'width'
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
    for word in sorted(old_matches.keys()):
        old_match_candidates = old_matches[word]
        new_match_candidates = new_matches[word]

        # Product between every token in image and every token in template
        # that has the same word
        for old_candidate, new_candidate in itertools.product(old_match_candidates, new_match_candidates):
            cv_match = cv2.DMatch(
                _distance=pixel_distance(old_candidate["nposition"], new_candidate["nposition"]),
                _queryIdx=len(matches),
                _trainIdx=len(matches),
                _imgIdx=0,
            )

            matches.append(cv_match)

            # Keypoints on old_img
            kp_old = cv2.KeyPoint(
                x=old_size["width"]
                * (old_candidate["nposition"]["left"] + old_candidate["nposition"]["right"])
                / 2,
                y=old_size["height"]
                * (old_candidate["nposition"]["bottom"] + old_candidate["nposition"]["top"])
                / 2,
                size=1,
            )
            kpsA.append(kp_old)

            # Keypoints on new_img
            kp_new_candidate = cv2.KeyPoint(
                x=new_size["width"]
                * (new_candidate["nposition"]["left"] + new_candidate["nposition"]["right"])
                / 2,
                y=new_size["height"]
                * (new_candidate["nposition"]["bottom"] + new_candidate["nposition"]["top"])
                / 2,
                size=1,
            )
            kpsB.append(kp_new_candidate)

    return (kpsA, kpsB), matches


def keypoints(
    *,
    config,
    matches,
    kps,
    old_image=None,
    new_image=None,
    debug=False,
):
    """Analyze images with ORB and RANSAC.
    Parameters:
        config (dict): config parameters
        old_image: cv2 image object to be analyzed
        new_image: cv2 image object to be analyzed
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
        matchedVis = cv2.drawMatches(old_image, kpsA, new_image, kpsB, matches, None)
        cv2.imwrite("matched-keypoints.png", matchedVis)

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
            old_image, kpsA, new_image, kpsB, successful_matches, None
        )
        cv2.imwrite("successful-matched-keypoints.png", matchedVis)

    return H, mask


def approx_match(old, new):
    new_normed = new['text'].lower().strip(" ,.-")
    old_normed = old['text_lower'].strip(' ,.-')
    return new_normed == old_normed or (new_normed in old_normed) or (old_normed in new_normed)

def extract_text_spans_from_box(
    expansion, new_width, new_height, new_tokens, box, debug
):
    """
    Given a box on a template, extract image tokens that lie within the
    box.
    Parameters:
        new_width: width of old page
        new_height: height of new page
        new_tokens (dict list): list of tokens on input image, as returned
            from OCR
        box (dict list): box reprojected onto new page
    """
    # TODO: Fix this O(N^2) operation with something like a KD-Tree
    toks_in_box = list(
        # CONSIDERING USING ATOL BASED ON SIZE OF TOKEN TO ACCOUNT FOR VARIATION IN FONT SIZE
        filter(
            lambda tok: in_box(
                tok["position"],
                box["position"],
                atol=expansion,
            ),
            new_tokens,
        )
    )

    if not toks_in_box:
        # Relax distance criteria text match is exact
        toks_in_box = list(
            filter(
                lambda tok: in_box(
                    tok["position"],
                    box["position"],
                    atol=expansion * 3,
                ) and approx_match(box, tok),
                new_tokens,
            )
        )

    toks = sorted(toks_in_box, key=lambda t: t['doc_offset']["start"])
    text_spans = order_text_spans_from_zone(box, toks)

    return {
        "label": box["label"],
        "text_spans": text_spans,
        "page_num": box["page_num"],
        "input_length": len(box['text_lower']),
        "output_length": len("".join(t['text'] for t in toks_in_box)),
        "before": box['text_lower'].strip(),
        "after": "".join(t['text'].lower().strip() for t in toks_in_box)
    }


def order_text_spans_from_zone(box, toks):
    """
    Given the extracted tokens from a box, orders tokens by position and
    returns the GlobalTokenSpan information (start, end, page_num)
    for each piece of text, as well as the joined text string
    """
    if len(toks) == 0:
        return []

    # Return start/end indices for each group of tokens. If tokens are
    # contiguous, join them
    text_spans = []
    prev_end = None
    for token in toks:
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

    return text_spans


def merge_adjacent_spans(tokens):
    """
    {
        "token": {
            'label': 'a',
            ...
        },
        "spans": {
            "text_spans": [
                {
                    "start": 0,
                    "end": 1,
                    "page_num": 0
                },
            ...
            ]
        },
        "original_token": {
            'label': 'a',
            ...
        }
    )
    """
    tokens = sorted([t for t in tokens if t['spans']['text_spans']], key=lambda t: t['spans']['text_spans'][0]['start'])
    if not len(tokens):
        return tokens

    merged_tokens = [tokens[0]]
    for token in tokens[1:]:
        prev_token = merged_tokens[-1]
        if token["token"]["label"] != prev_token["token"]["label"]:
            merged_tokens.append(token)
        elif (token['spans']['text_spans'][0]['start'] - 2) <= prev_token['spans']['text_spans'][-1]['end']:
            # Can happen due to reading order oddities -- rare but required to prevent hard error on apply_labgels.py
            prev_token['spans']['text_spans'][-1]['end'] = max(token['spans']['text_spans'][-1]['end'], prev_token['spans']['text_spans'][-1]['end'])
        else:
            merged_tokens.append(token)
    return merged_tokens


def visualize_tokens(image_path, tokens, filename):
    original_image = cv2.imread(image_path)
    for token in tokens:
        p = token['position']
        cv2.rectangle(
            original_image, 
            (p['left'], p['top']), 
            (p['right'], p['bottom']),
            color=(255, 0, 0)
        )
    print(f"Saving to: {filename}")
    cv2.imwrite(filename, original_image)


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
    old_labels = json.loads(old_ocr_for_file["labels"].values[0])

    label_to_token_by_page = {}
    for p_0, n_0 in zip(old_ocr, new_ocr):
        old_page_ocr = p_0["pages"][0]
        page_number = old_page_ocr["page_num"]
        try:
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
            old_match, new_match = get_word_match_dict(
                norm_old_points, norm_new_points, matches
            )

            kps, cv_matches = get_keypoints(old_match, new_match, old_size, new_size)

            original_image = cv2.imread(old_images[page_number])
            new_image = cv2.imread(new_images[page_number])
            homography, mask = keypoints(
                config=aligner_config,
                kps=kps,
                matches=cv_matches,
                new_image=new_image,
                old_image=original_image,
                debug=debug,
            )
            match_ratio = mask.sum() / len(mask) if mask is not None else 0.0
            if match_ratio < aligner_config.min_keypoint_match_ratio:
                label_to_token_by_page[page_number] = []
                print(f"Failed to align {file} page {page_number}")
                continue

            if homography is None:
                message = f"RANSAC failed for {file} on page {page_number}, using identity matrix."
                print(message)
                logging.warning(message)
                affine_warp = np.identity(3)
            else:
                affine_warp = np.vstack([homography, [0, 0, 1]])

            ex_old_tokens, old_tokens_new_coords = transform_and_extract(
                old_labels, min_offset, max_offset, old_tokens, affine_warp
            )

            label_token_map = defaultdict(list)
            error_by_setting = defaultdict(list)

            # No source labels on this page
            if not ex_old_tokens.extracted_tokens:
                label_to_token_by_page[page_number] = []
                continue

            for expansion_in_pixels in aligner_config.TEXT_EXTRACTION_ATOL:
                for old_token_new_coords, o in zip(
                    old_tokens_new_coords, ex_old_tokens.extracted_tokens
                ):
                    extracted_spans_for_label = extract_text_spans_from_box(
                        expansion_in_pixels,
                        new_size["width"],
                        new_size["height"],
                        norm_new_tokens,
                        old_token_new_coords,
                        debug=aligner_config.debug,
                    )
                    label_token_map[expansion_in_pixels].append(
                        {
                            "token": old_token_new_coords,
                            "spans": extracted_spans_for_label,
                            "original_token": o,
                        }
                    )
                    error_by_setting[expansion_in_pixels].append(
                        abs(
                            extracted_spans_for_label["input_length"]
                            - extracted_spans_for_label["output_length"]
                        )
                    )

            for k, v in error_by_setting.items():
                error_by_setting[k] = np.mean(v)

            best_setting = min(error_by_setting.items(), key=lambda x: x[1])[0]

            best_map = label_token_map[best_setting]

            if aligner_config.debug:
                for mapped_token in best_map:
                    before = mapped_token["spans"]["before"]
                    after = mapped_token["spans"]["after"]
                    if len(before) != len(after):
                        print(before, "-->", after)

            for token in best_map:
                for span in token["spans"]["text_spans"]:
                    assert span["start"] <= span["end"], f"Before merge: {token}"

            label_token_map = merge_adjacent_spans(best_map)

            for token in best_map:
                for span in token["spans"]["text_spans"]:
                    assert span["start"] <= span["end"], f"After merge: {token}"

            label_to_token_by_page[page_number] = label_token_map
        except:
            print(f"Failed to align {file} page {page_number}")
            label_to_token_by_page[page_number] = []

    return label_to_token_by_page


def transform_and_extract(labels, min_offset, max_offset, old_tokens, affine_warp):
    labels_on_page = [
        l for l in labels if l["end"] <= max_offset and l["start"] >= min_offset
    ]
    ex_tokens = ExtractedTokens(labels_on_page)
    ex_tokens.collect_tokens(old_tokens, raise_for_no_match=True)
    transformed = transform_tokens(affine_warp, ex_tokens.extracted_tokens)
    return ex_tokens, transformed


def run_all_docs(config, k=None):
    logging.info("Loading new dataset...")
    new_directory = config.new_engine_folder_name
    new_ocr_path = f"{new_directory}/all_labels.csv"
    new_ocr_df = pd.read_csv(new_ocr_path)
    new_ocr_df["file_name"] = new_ocr_df["document_path"].apply(
        lambda x: os.path.basename(x)
    )
    new_file_names = set(new_ocr_df["file_name"])

    logging.info("Loading old dataset...")
    old_directory = config.old_engine_folder_name
    old_ocr_path = f"{old_directory}/all_labels.csv"
    old_ocr_df = pd.read_csv(old_ocr_path)
    old_ocr_df["file_name"] = old_ocr_df["document_path"].apply(
        lambda x: os.path.basename(x)
    )
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
    n_files = len(list(common_file_names))
    for idx, f in enumerate(sorted(list(common_file_names)[:k])):
        print(f"Processing file {idx + 1}/{n_files}: {f}")
        new_ocr_for_file = new_ocr_df[new_ocr_df["file_name"] == f]
        old_ocr_for_file = old_ocr_df[old_ocr_df["file_name"] == f]
        matched_labels_for_doc = run_all_pages_for_doc(
            f,
            new_ocr_for_file,
            old_ocr_for_file,
            config,
            debug=config.debug,
        )
        mappings_by_file_name[f] = matched_labels_for_doc

    return mappings_by_file_name


def run(config, num_docs=None, summary_file="./summary.xlsx"):
    with open(config, "r") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    aligner_config = AlignerConfig(config_dict)
    all_results = run_all_docs(aligner_config, k=num_docs)
    migrated_labels_csv = os.path.join(
        aligner_config.new_engine_folder_name, "revised_labels.json"
    )

    # old_ocr = pd.read_csv(
    #     f"./{aligner_config.old_engine_folder_name}/all_labels.csv", index_col=0
    # )
    # summary_by_file, overall_summary = summarize_results(all_results, old_ocr)
    # convert_to_excel(summary_by_file, overall_summary, summary_file)

    with open(migrated_labels_csv, "w") as fd:
        json.dump(all_results, fd)



if __name__ == "__main__":
    Fire(run)
