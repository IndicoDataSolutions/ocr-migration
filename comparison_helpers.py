from indico_toolkit.metrics.compare_ground_truth import CompareGroundTruth
from itertools import groupby
import json
import pandas as pd
import os


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
            "page_num": a["page_num"],
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
    # sourcery skip: inline-immediately-returned-variable
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


def collapse(preds):
    collapsed = []
    for index, grouped_preds in groupby(
        preds, key=lambda x: x["token"]["prediction_index"]
    ):
        to_list = list(grouped_preds)
        to_list.sort(key=lambda x: x["spans"]["text_spans"][0]["start"])
        collapsed.append(
            {
                "text": " ".join([t["spans"]["text"] for t in to_list]),
                "start": to_list[0]["spans"]["text_spans"][0]["start"],
                "end": to_list[-1]["spans"]["text_spans"][0]["end"],
                "label": to_list[0]["spans"]["label"],
                "page_num": to_list[0]["spans"]["page_num"],
            }
        )
    return collapsed


def filter_empty(preds):
    return filter(lambda x: x["spans"]["text_spans"], preds)


def summarize_results(results, labels_raw):
    records_by_file = {}
    overall_summary = []
    labels_raw["file_name"] = labels_raw["document_path"].apply(
        lambda x: os.path.basename(x)
    )

    for file_name in results.keys():
        all_transformed = results[file_name]
        flattened = []
        for _, a_t in all_transformed.items():
            flattened.extend(collapse(filter_empty(list(a_t))))
        original_labels = json.loads(
            labels_raw[labels_raw["file_name"] == file_name]["labels"].values[0]
        )
        missing_records = []
        matched_records = []
        all_label_names = list({o["label"] for o in original_labels})
        # flatten labels and page number
        summary_by_label = {o: [0, 0] for o in all_label_names}
        for o in original_labels:
            label = o["label"]
            text = o["text"]
            if matches := [
                t for t in flattened if t["label"] == label and t["text"] == text
            ]:
                summary_by_label[label][0] += len(matches)
                matched_records.append(
                    {
                        "original_label": o,
                        "page_number": matches[0]["page_num"],
                        "new_span": matches,
                    }
                )
            else:
                missing_records.append({"original_label": o})
                summary_by_label[label][1] += 1

        overall_summary.append((file_name, summary_by_label))

        records_by_file[file_name] = {
            "matched": matched_records,
            "missing": missing_records,
        }

    condensed = []
    for f, by_label in overall_summary:
        total_matched_labels = sum(res[0] for l, res in by_label.items())
        total_labels = sum(res[0] + res[1] for l, res in by_label.items())
        by_label = {l: res[0] / (res[0] + res[1]) for l, res in by_label.items()}
        by_label["Total"] = total_matched_labels / total_labels
        condensed.append((f, by_label))
    overall_summary_df = pd.DataFrame(
        condensed, columns=["file_name", "results"]
    ).set_index("file_name")
    overall_summary_df = pd.concat(
        [overall_summary_df, overall_summary_df["results"].apply(pd.Series)], axis=1
    ).drop("results", axis=1)

    return records_by_file, overall_summary_df


def convert_to_excel(results_dict, summary_df, out_file):
    writer = pd.ExcelWriter(out_file, engine="xlsxwriter")
    summary_df.to_excel(writer, sheet_name="Overall Summary")
    for file_name, results in results_dict.items():
        df = pd.DataFrame.from_records(
            [r["original_label"] for r in results["missing"]], index=None
        )
        df.to_excel(writer, sheet_name=file_name[:30], index=None)
    writer.close()
