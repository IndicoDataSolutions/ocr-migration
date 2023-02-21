from indico_toolkit.metrics.compare_ground_truth import CompareGroundTruth


def convert_results(res):
    all_spans = [r["spans"] for r in res]
    all_spans = [dict(item, **{"checked": False}) for item in all_spans]
    all_converted = []
    for a in all_spans:
        if a["checked"]:
            continue
        converted = {
            "label": a["label"],
            "text": a["text"],
            "start": a["text_spans"][0]["start"],
            "end": a["text_spans"][0]["end"],
        }
        adjacent = True
        last_end = a["text_spans"][0]["start"] - 1
        next_start = a["text_spans"][0]["end"] + 1
        while adjacent:
            left_adjacent = [
                s
                for s in all_spans
                if s["text_spans"][0]["end"] == last_end and s["label"] == a["label"]
            ]
            right_adjacent = [
                s
                for s in all_spans
                if s["text_spans"][0]["start"] == next_start
                and s["label"] == a["label"]
            ]
            if left_adjacent:
                converted["text"] = left_adjacent[0]["text"] + " " + converted["text"]
                converted["start"] = left_adjacent[0]["text_spans"][0]["start"]
                a["checked"] = True
                left_adjacent[0]["checked"] = True
            if right_adjacent:
                converted["text"] += " " + right_adjacent[0]["text"]
                converted["end"] = right_adjacent[0]["text_spans"][0]["end"]
                a["checked"] = True
                right_adjacent[0]["checked"] = True

            adjacent = left_adjacent or right_adjacent
            last_end = converted["start"] - 1
            next_start = converted["end"] + 1
        all_converted.append(converted)

    return all_converted


def convert_for_file(labels, results_for_file):
    # all_converted = convert_toolkit(results_for_file)
    all_converted = []
    for page_num, res in results_for_file.items():
        all_converted.extend(convert_results(res))

    gt_compare = CompareGroundTruth(ground_truth=labels, predictions=all_converted)
    gt_compare.set_all_label_metrics("overlap")
    gt_compare.set_overall_metrics()
    final = {
        "overall": gt_compare.overall_metrics,
        "by_label": gt_compare.all_label_metrics,
    }
    return final
