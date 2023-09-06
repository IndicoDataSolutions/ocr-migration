from indico import IndicoClient, IndicoConfig
from indico.queries import (
    CreateWorkflow,
    GetWorkflow,
    AddModelGroupComponent,
    GetDataset,
)
from indico.types import NewLabelsetArguments, ModelTaskType
import os
import pandas as pd
import fire
import json
import logging
import tqdm
import time

from enum import Enum

from get_datasets import GraphQLMagic


class ModelTaskType(Enum):
    """A list of valid task types for a model group."""
    ANNOTATION = 6


class GetLabelsetMeta(GraphQLMagic):
    query = """
    query GetTargetNames($datasetId:Int!){
        dataset(id:$datasetId) {
            labelsets {
                id
                name
                targetNames {
                    id
                    name
                }
            }
        }
    }
    """


class GetQuestionnaireMeta(GraphQLMagic):
    query = """
    query GetQuestionnaireDatafiles($questionnaireId: Int!){
        questionnaire(id: $questionnaireId){
            examples(numExamples:1000) {
                id
                datafileId
            }
        }
    }
    """


class ModelGroupMeta(GraphQLMagic):
    query = """
    query GetModelGroupMeta($modelGroupId: Int!){
        modelGroup(modelGroupId: $modelGroupId) {
            labelsetColumnId
            pagedExamples(limit: 1000) {
                examples {
                    id
                    datafileId
                }
            }
        }
    }
    """


class SubmitLabel(GraphQLMagic):
    query = """
    mutation SubmitLabel($labelsetId: Int!, $labels: [LabelInput]!){
        submitLabelsV2(
            labelsetId: $labelsetId,
            labels: $labels,
        ){
            success
        }
    }
    """


logging.basicConfig(level=os.getenv("LOGGING_LEVEL", "DEBUG"))


def get_target_names(revised_labels):
    target_names = set()
    for filename, labels in revised_labels.items():
        for str_page_num, page_labels in labels.items():
            for label in page_labels:
                target_names.add(label["spans"]["label"])
    return target_names


def reformat_labels(label_data_by_page, cls_map):
    reformatted = []
    for str_page_num, page_labels in label_data_by_page.items():
        for label in page_labels:
            for text_span in label["spans"]["text_spans"]:
                text_span["pageNum"] = text_span.pop("page_num")
                reformatted.append(
                    {"clsId": cls_map[label["spans"]["label"]], "spans": [text_span]}
                )
    return reformatted


def apply_labels(
    new_dataset_id,
    label_json_path="./new/revised_labels.json",
    new_export_path="./new/raw_export.csv",
    workflow_name=None,
    workflow_id=None,
    mg_name=None,
    mg_id=None,
    datacolumn_name="document",
    host="app.indico.io",
    api_token_path="prod_api_token.txt",
):
    print("Reading csv...")
    new_df = pd.read_csv(new_export_path)
    name_to_file_id = {row["file_name"]: row["file_id"] for _, row in new_df.iterrows()}
    client = IndicoClient(config=IndicoConfig(host=host, api_token_path=api_token_path))
    print("Getting dataset details...")
    dataset = client.call(GetDataset(id=new_dataset_id))
    print("Loading revised labels...")
    revised_labels = json.load(open(label_json_path))

    print("Fetching target name info...")
    target_names = get_target_names(revised_labels)

    if not workflow_name and not workflow_id:
        raise ValueError("Must provide either workflow_name or workflow_id")

    if not mg_name and not mg_id:
        raise ValueError("Must provide either mg_name or mg_id")

    if workflow_name:
        logging.info("Creating workflow...")
        workflow = client.call(
            CreateWorkflow(name=workflow_name, dataset_id=new_dataset_id)
        )
    else:
        print("Fetching workflow details...")
        workflow = client.call(GetWorkflow(workflow_id=workflow_id))

    # TODO: pass as arg
    after_component_id = workflow.component_by_type("INPUT_OCR_EXTRACTION").id

    if mg_name:
        logging.info("Creating model group component...")
        workflow = client.call(
            AddModelGroupComponent(
                name=mg_name,
                dataset_id=new_dataset_id,
                after_component_id=after_component_id,
                source_column_id=dataset.datacolumn_by_name(datacolumn_name).id,
                new_labelset_args=NewLabelsetArguments(
                    name=mg_name,
                    task_type=ModelTaskType.ANNOTATION,
                    datacolumn_id=dataset.datacolumn_by_name(datacolumn_name).id,
                    target_names=list(set(target_names)),
                ),
                workflow_id=workflow.id,
            )
        )
        time.sleep(60)

    matching_component = next(
        c
        for c in workflow.components
        if c.component_type == "MODEL_GROUP"
        and (c.model_group.name == mg_name or c.model_group.id == mg_id)
    )
    model_group_id = matching_component.model_group.id
    questionnaire_id = matching_component.model_group.questionnaire_id
    model_group_meta = client.call(ModelGroupMeta(modelGroupId=model_group_id))['modelGroup']
    labelset_id = model_group_meta['labelsetColumnId']
    examples = model_group_meta['pagedExamples']['examples'] 

    print("Fetching label / questionnaire metadata...")
    questionnaire = client.call(GetQuestionnaireMeta(questionnaireId=questionnaire_id))['questionnaire']
    labelset_meta = client.call(GetLabelsetMeta(datasetId=new_dataset_id))
    labelset = next(
        lset
        for lset in labelset_meta["dataset"]["labelsets"]
        if lset["id"] == labelset_id
    )
    examples += questionnaire["examples"]
    cls_map = {tname["name"]: tname["id"] for tname in labelset["targetNames"]}
    file_id_to_example_id = {
        example["datafileId"]: example["id"] for example in examples
    }
    for filename, label_data in tqdm.tqdm(revised_labels.items()):
        print("Processing", filename)
        file_id = name_to_file_id[filename]
        example_id = file_id_to_example_id[file_id]
        targets = reformat_labels(label_data, cls_map)
        if not targets:
            print(f"No targets found for {filename}, skipping...")
            continue
        client.call(SubmitLabel(labelsetId=labelset_id, labels=[{'exampleId': example_id, 'targets': targets, 'override': True}]))

if __name__ == "__main__":
    fire.Fire(apply_labels)
