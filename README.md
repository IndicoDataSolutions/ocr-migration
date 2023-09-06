# ocr-migration

### Scripts for Migrating Labeled Data from one OCR engine to another

**First, run [get_datasets.py](https://github.com/IndicoDataSolutions/ocr-migration/blob/main/get_datasets.py) to gather OCR + Labels for the original dataset.**

```bash
python3 get_datasets.py --name old --dataset_id 12960 --labelset_id 27412 --text_col="document" --host="your_url" --api_token_path="../my_indico_api_token.txt"
```

You should end up with file structure like this:

- /old:
  - /files/
  - /images/
  - /jsons/
  - /all_labels.csv
  - /raw_export.csv

If you don't know the text_col name you can get it by look at the dataset list page in Indico (for any non-CSV/txt 
uploaded dataset it will default to be "document") or from the `old/raw_export.csv` file that is downloaded on first 
run (it will hit an exception when those columns don't exist, and you can re-run the script with the right arguments).

If you don't have the labelset ID handy, that can be found using the GraphQL query below (substitute your own dataset ID):

```
{
  dataset(id:1234) {
    labelsets {
      id
      name
      taskType
    }
  }
}
```

If the script fails for any reason you should be safe to re-run as the files are cached -- but it will still take time to load page JSONs from disk (several minutes for all page files in a ~200 doc dataset).

**Next, upload the files in the `old/files` directory to a new dataset using the UI**

You'll want to use the files in this directory even if you have access to the originals you uploaded, to guarantee the file names match up.
Make sure to configure the dataset to use the OCR you're looking to migrate to.  
You'll want to make note of the dataset ID because you'll need it in the next step.

**After this is complete, run the `get_datasets.py script again.**

This time you won't need to supply
arguments related to labels, because we only have raw docs.

```bash
python3 get_datasets.py --name new --dataset_id 13431 --text_col="document" --host="your_url" --api_token_path="../my_indico_api_token.txt"
```

This will produce the new folder which contains the migrated labels:

- /new:
  - /files/
  - /images/
  - /jsons/
  - /all_labels.csv
  - /raw_export.csv

**Next, run ocr_migration.py.**

- ocr_migration.py reads in "migration_config.yaml" to set the configuration of the migration-- any settings changes can be made 
within that file.
- the location where an excel summary should be saved is an optional argument, defaults to "./summary.xlsx".

```
python3 ocr_migration.py 
```

If you want to test on a subset you can pass the num_docs argument

```
python3 ocr_migration.py --config "old_to_new.yaml" --num_docs 1
```

**Finally, you can apply the revised labels to your new dataset.**

Be warned that this will overwrite any existing labels on that labelset.

If you used the default values above then you can skip entering in paths to your labels/export 
(e.g. --label_json_path new/revised_labels.json --new_export_path new/raw_export.csv)

Simply use your new dataset ID and name your new worklfow and model group
```
python3 apply_labels.py 13431 --workflow_name "Workflow converted to ReadAPI" --mg_name="my model group" --host="your_url" --api_token_path="../my_indico_api_token.txt"
```

If you already have a workflow and/or modelgroup you want to apply the labels to then, include those values in place of names (this would be unusual):

```
python3 apply_labels.py 13431 --workflow_id 5646 -- mg_id 2432
```

# TODO:

- Merge spans that can be merged so that it looks cleaner in the UI
