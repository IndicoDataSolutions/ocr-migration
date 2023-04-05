from indico import IndicoClient, IndicoConfig
from indico.queries import CreateWorkflow, GetWorkflow, AddModelGroupComponent, GetDataset
from indico.queries.storage import UploadBatched, CreateStorageURLs
from indico.types import NewLabelsetArguments, ModelTaskType
import os
import pandas as pd
import fire
import json
import logging
import tqdm

from indico.client import GraphQLRequest

class GraphQLMagic(GraphQLRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(query=self.query, variables=kwargs)


class GetTargetNames(GraphQLMagic):
    query = """
    query GetTargetNames($datasetId:Int!){
        dataset(id:$datasetId) {
            labelsets {
                id
                targetNames {
                    id
                    name
                }
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
                target_names.add(label['spans']['label'])
    return target_names


def reformat_labels(label_data_by_page, cls_map):
    reformatted = []
    for str_page_num, page_labels in label_data_by_page.items():
        for label in page_labels:
            span_data = label['token']['doc_offset']
            span_data['pageNum'] = label['token']['page_num']
            reformatted.append({'clsId': cls_map[label['spans']['label']], 'spans': [span_data]})
    return reformatted

    
def apply_labels(
    label_json_path, 
    new_dataset_id, 
    new_export_path,
    workflow_name=None, 
    workflow_id=None, 
    mg_name=None, 
    mg_id=None, 
    datacolumn_name="document",
    host="app.indico.io", 
    api_token_path="prod_api_token.txt", 
):
    new_df = pd.read_csv(new_export_path)
    name_to_file_id = {row['file_name']: row['file_id'] for _, row in new_df.iterrows()}
    client = IndicoClient(config=IndicoConfig(host=host, api_token_path=api_token_path))
    dataset = client.call(GetDataset(id=new_dataset_id))
    revised_labels = json.load(open(label_json_path))

    target_names = get_target_names(revised_labels)

    if not workflow_name and not workflow_id:
        raise ValueError("Must provide either workflow_name or workflow_id")
    
    if not mg_name and not mg_id:
        raise ValueError("Must provide either mg_name or mg_id")


    if workflow_name:
        logging.info("Creating workflow...")
        workflow = client.call(CreateWorkflow(name=workflow_name, dataset_id=new_dataset_id))
    else:
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
                    target_names=list(set(target_names))
                ),
                workflow_id=workflow.id,
            )
        )
    matching_component = next(
        c for c in workflow.components if c.component_type == "MODEL_GROUP"
        and (c.model_group.name == mg_name or c.model_group.id == mg_id)
    )
    model_group_id = matching_component.model_group.id
    model_group_meta = client.call(ModelGroupMeta(modelGroupId=model_group_id))['modelGroup']
    labelset_id = model_group_meta['labelsetColumnId']
    examples = model_group_meta['pagedExamples']['examples']
    labelset = next(
        lset for lset in client.call(GetTargetNames(datasetId=new_dataset_id))['dataset']['labelsets'] 
        if lset['id'] == labelset_id
    )
    cls_map = {
        tname['name']: tname['id'] for tname in labelset['targetNames']
    }
    file_id_to_example_id = {
        example['datafileId']: example['id'] for example in examples
    }
    for filename, label_data in tqdm.tqdm(revised_labels.items()):
        file_id = name_to_file_id[filename]
        example_id = file_id_to_example_id[file_id]
        targets = reformat_labels(label_data, cls_map)
        client.call(SubmitLabel(labelsetId=labelset_id, labels=[{'exampleId': example_id, 'targets': targets, 'override': True}]))

if __name__ == "__main__":
    fire.Fire(apply_labels)