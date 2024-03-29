import glob
import os
import pandas as pd
import traceback
import json
import fire
import os
import logging
import uuid
import tqdm

logging.basicConfig(level=os.getenv("LOGGING_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from indico import IndicoClient, IndicoConfig
from indico.errors import IndicoError
from indico.queries import (
    RetrieveStorageObject,
    GetDataset,
    DownloadExport,
    CreateExport,
)
from indico.client import GraphQLRequest


class GraphQLMagic(GraphQLRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(query=self.query, variables=kwargs)


class GetDatafileIDs(GraphQLMagic):
    query = """
    query getDatafileIDs($datasetId: Int!){
        dataset(id: $datasetId) {
            files {
                fileType
                id
                name
                rainbowUrl
            }
        }
    }
    """


class GetDatafileByID(GraphQLMagic):
    query = """
    query getDatafileById($datafileId: Int!) {
        datafile(datafileId: $datafileId) {
            pages {
                id
                pageInfo
                image
                pageNum
            }
        }
    }
    """


class GetLabelsetName(GraphQLMagic):
    query = """
    query GetTargetNames($datasetId:Int!){
        dataset(id:$datasetId) {
            labelsets {
                id
                name
            }
        }
    }
    """


def get_export(client, dataset_id, labelset_id=None):
    # Get dataset object
    dataset = client.call(GetDataset(id=dataset_id))

    if labelset_id is None and dataset.labelsets:
        labelset_id = dataset.labelsets[0].id

    if labelset_id is not None:
        # Create export object using dataset's id and labelset id
        logger.info("Creating export using Indico API...")
        export = client.call(
            CreateExport(
                dataset_id=dataset.id,
                labelset_id=labelset_id,
                file_info=True,
                wait=True,
            )
        )

        # Use export object to download as pandas csv
        logging.info("Downloading export...")
        df = client.call(DownloadExport(export.id))
        df = df.rename(columns=lambda col: col.rsplit("_", 1)[0])
    else:
        df = generate_fake_export_sans_labels(client, dataset_id)

    return df


def generate_fake_export_sans_labels(client, dataset_id):
    """
    Get text of each doc and convert to a pd.DataFrame
    """
    datafiles = client.call(GetDatafileIDs(datasetId=dataset_id))["dataset"]["files"]
    records = []
    for datafile in datafiles:
        records.append(
            {
                "file_id": datafile["id"],
                "file_name": datafile["name"],
                "file_url": datafile["rainbowUrl"],
            }
        )
    return pd.DataFrame.from_records(records)


def text_from_ocr(page_ocrs):
    return "\n".join(page["pages"][0]["text"] for page in page_ocrs)


def reformat_labels(labels, document):
    spans_labels = json.loads(labels)
    old_labels_i = []
    for target in spans_labels["targets"]:
        old_labels_i.append(
            {
                "label": target["label"],
                "start": min(l["start"] for l in target["spans"]),
                "end": max(l["end"] for l in target["spans"]),
            }
        )
        old_labels_i[-1]["text"] = document[
            old_labels_i[-1]["start"] : old_labels_i[-1]["end"]
        ]
    return json.dumps(old_labels_i)


def get_ocr_by_datafile_id(client, datafile_id, dataset_dir, filename):
    """
    Given an Indico client and a datafile ID, download OCR data for all pages
    along with page image PNGs for each page.
    """
    datafile_meta = client.call(GetDatafileByID(datafileId=datafile_id))
    page_ocrs, page_images = [], []
    filename = filename.strip()
    local_page_image_dir = os.path.join(dataset_dir, "images", filename)
    local_page_json_dir = os.path.join(dataset_dir, "jsons", filename)
    os.makedirs(local_page_image_dir, exist_ok=True)
    os.makedirs(local_page_json_dir, exist_ok=True)
    for page in datafile_meta["datafile"]["pages"]:
        page_info = page["pageInfo"]
        page_json_file = os.path.join(
            local_page_json_dir, f"page_{page['pageNum']}.json"
        )
        page_image_file = os.path.join(
            local_page_image_dir, f"page_{page['pageNum']}.png"
        )
        if os.path.exists(page_json_file):
            page_ocr = json.load(open(page_json_file))
        else:
            page_ocr = client.call(RetrieveStorageObject(page["pageInfo"]))
        # Could just return page image and save to file in inner loop if required
        if not os.path.exists(page_image_file):
            page_image = client.call(RetrieveStorageObject(page["image"]))
            with open(page_image_file, "wb") as fd:
                fd.write(page_image)
        page_ocrs.append(page_ocr)
        page_images.append(page_image_file)
    return page_ocrs, page_images


def get_dataset(
    name,
    dataset_id,
    labelset_id=None,
    label_col="labels",
    text_col="document",
    filename_col="file_name",
    host="app.indico.io",
    api_token_path="prod_api_token.txt",
):
    # TODO: Get label col name from labelset metadata
    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, "images"), exist_ok=True)
    os.makedirs(os.path.join(name, "files"), exist_ok=True)
    my_config = IndicoConfig(
        host=host,
        api_token_path=api_token_path,
    )
    client = IndicoClient(config=my_config)
    if labelset_id:
        labelset = next(
            labelset
            for labelset in client.call(GetLabelsetName(datasetId=dataset_id))[
                "dataset"
            ]["labelsets"]
            if labelset["id"] == labelset_id
        )
        label_col = labelset["name"]

    export_path = os.path.join(name, "raw_export.csv")

    if not os.path.exists(export_path):
        raw_export = get_export(client, dataset_id, labelset_id)
        raw_export.to_csv(export_path)
    else:
        raw_export = pd.read_csv(export_path)

    records = raw_export.to_dict("records")
    output_records = []
    label_col = label_col.rsplit("_", 1)[0]

    for i, row in enumerate(tqdm.tqdm(records)):
        filename = os.path.splitext(os.path.basename(row[filename_col]))[0]
        document_path = os.path.join(
            name, "files", filename + "." + row["file_name"].split(".")[-1]
        )
        page_ocrs, page_image_paths = get_ocr_by_datafile_id(
            client, row["file_id"], dataset_dir=name, filename=filename
        )

        # Try to get text from export, but fallback to reconstructing from page OCR
        if text_col in row:
            text = row[text_col]
        else:
            text = text_from_ocr(page_ocrs)

        # DF doesn't have labels or labels are null for a file
        if label_col not in row or pd.isna(row[label_col]):
            labels = None
        else:
            labels = reformat_labels(row[label_col], text)

        output_record = {"ocr": json.dumps(page_ocrs), "text": text, "labels": labels}
        output_record["image_files"] = json.dumps(page_image_paths)
        output_record["document_path"] = document_path

        with open(document_path, "wb") as fp:
            fp.write(client.call(RetrieveStorageObject(row["file_url"])))

        output_records.append(output_record)

    csv_path = os.path.join(name, "all_labels.csv")
    logger.info("Creating CSV...")
    pd.DataFrame.from_records(output_records).to_csv(csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(get_dataset)
