from indico import IndicoClient, IndicoConfig
from indico.queries import CreateDataset
from indico.queries.storage import UploadBatched, CreateStorageURLs
import os
import pandas as pd
import json


def make_dataset_for_upload(client, document_dir, migrated, dataset_name):
    doc_paths = []
    labels = []

    for file_name, migration_result_for_file in migrated.items():
        document_path = os.path.join(document_dir, file_name)
        doc_paths.append(document_path)
        labels.append(migration_result_for_file)

    urls = client.call(
        UploadBatched(doc_paths, batch_size=10, request_cls=CreateStorageURLs)
    )

    csv_path = f"{dataset_name}.csv"
    pd.DataFrame({"pdf_link": urls, "annotation": labels}).to_csv(csv_path, index=False)
    return csv_path


def upload(
    host, api_token_path, document_dir, dataset_name, migrated_labels, **ocr_options
):
    client = IndicoClient(config=IndicoConfig(host=host, api_token_path=api_token_path))
    csv_path = make_dataset_for_upload(
        client, document_dir, migrated_labels, dataset_name
    )

    dataset = client.call(
        CreateDataset(
            name=dataset_name,
            files=[csv_path],
            dataset_type="DOCUMENT",
        )
    )

    return dataset
