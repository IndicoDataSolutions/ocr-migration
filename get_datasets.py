import glob
import pandas as pd
import traceback
import json
import fire
import os
import uuid

from sklearn.model_selection import train_test_split

from indico import IndicoClient, IndicoConfig
from indico.errors import IndicoError
from indico.queries import (
    RetrieveStorageObject,
    GetDataset,
    DownloadExport, CreateExport
)
from indico.client import GraphQLRequest

class GraphQLMagic(GraphQLRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(query=self.query, variables=kwargs)

class GetDatafileByID(GraphQLMagic):
    query = """
    query getDatafileById($datafileId: Int!) {
        datafile(datafileId: $datafileId) {
            pages {
            id
            pageInfo
            image
            }
        }
    }
    """

def get_export(client, dataset_id, labelset_id):
    # Get dataset object
    dataset = client.call(GetDataset(id=dataset_id))
    
    # Create export object using dataset's id and labelset id
    export = client.call(
        CreateExport(
            dataset_id=dataset.id,
            labelset_id=dataset.labelsets[0].id,
            file_info=True,
            wait=True
        )
    )

    # Use export object to download as pandas csv
    csv = client.call(DownloadExport(export.id))
    csv = csv.rename(columns=lambda col: col.rsplit("_", 1)[0])
    return csv

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


def get_ocr_by_datafile_id(client, datafile_id):
    """
    Given an Indico client and a datafile ID, download OCR data for all pages
    along with page image PNGs for each page.
    """
    datafile_meta = client.call(GetDatafileByID(datafileId=datafile_id))
    page_ocrs, page_images = [], []
    for page in datafile_meta['datafile']['pages']:
        page_info = client.call(RetrieveStorageObject(page['pageInfo']))
        # Could just return page image and save to file in inner loop if required
        page_image = client.call(RetrieveStorageObject(page['image']))
        page_ocrs.append(page_info)
        page_images.append(page_image)
    return page_ocrs, page_images

def get_dataset(name, dataset_id, labelset_id, label_col="labels", text_col="text", host="app.indico.io", api_token_path="/home/m/api_keys/prod_api_token.txt"):
    if not os.path.exists(name):
        os.mkdir(name)
        os.mkdir(os.path.join(name, "images"))
        os.mkdir(os.path.join(name, "files"))


    my_config = IndicoConfig(
        host=host, api_token_path=api_token_path,
    )
    client = IndicoClient(config=my_config)

    export_path = os.path.join(name, "raw_export.csv")

    if not os.path.exists(export_path):
        raw_export = get_export(client, dataset_id, labelset_id)
        raw_export.to_csv(export_path)
    else:
        raw_export = pd.read_csv(export_path)

    records = raw_export.to_dict("records")
    output_records = []

    for i, row in enumerate(records):
        if pd.isna(row[label_col]):
            print("No labels - skipping")
            continue
        filename = str(uuid.uuid4())
        file_dir = os.path.join(name, "images", filename)
        os.makedirs(file_dir, exist_ok=True)
        local_page_pattern = os.path.join(file_dir, "page_{}.png")
        page_ocrs, page_images = get_ocr_by_datafile_id(client, row['file_id'])
        output_record = {
            "ocr": json.dumps(page_ocrs),
            "text": row[text_col],
            "labels": reformat_labels(row[label_col], row[text_col]),
        }
        image_files = []
        for (page_ocr, page_image) in zip(page_ocrs, page_images):
            local_page_image = local_page_pattern.format(page_ocr["pages"][0]["page_num"])
            image_files.append(local_page_image)
            with open(local_page_image, "wb") as fp:
                fp.write(page_image)
        output_record["image_files"] = json.dumps(image_files)
        document_path = os.path.join(name, "files", filename + "." + row["file_name"].split(".")[-1])
        output_record["document_path"] = document_path
        with open(document_path, "wb") as fp:
            fp.write(client.call(RetrieveStorageObject(row["file_url"])))
        output_records.append(output_record)

    # TODO: consider removing train test split logic
    train_records, test_val_records = train_test_split(output_records, test_size=0.4)
    test_records, val_records = train_test_split(test_val_records, test_size=0.5)
    for split, records in [
        ("train", train_records),
        ("test", test_records),
        ("val", val_records),
    ]:
        pd.DataFrame.from_records(records).to_csv(os.path.join(name, "{}.csv".format(split)))
        
    
if __name__ == "__main__":
    fire.Fire(get_dataset)
