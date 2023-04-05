# ocr-migration

### Scripts for Migrating Labeled Data from one OCR engine to another

**First, run [get_datasets.py](https://github.com/IndicoDataSolutions/ocr-migration/blob/main/get_datasets.py) to gather OCR + Labels for the original dataset. **

```bash
python3 get_datasets.py --name old --dataset_id 11723 --labelset_id 28306 --label_col="annotation" --text_col="pdf"
```

If you don't know the label_col or text_col name you can get them from the `old/raw_export.csv` file that is downloaded on first run (it will hit an exception when those columns don't exist, and you can re-run the script with the right arguments).

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

If something goes wrong the first time around, it's safe to re-run the script -- it will skip over files that
have already been successfully downloaded so it should be quicker.

Next, upload the files in the `old/files` directory to a new dataset and configure the dataset to use the OCR you're looking to migrate to. After this is complete, run the `get_datasets.py script again. This time you won't need to supply
arguments related to labels, because we only have raw docs.

```bash
python3 get_datasets.py --name new --dataset_id 13431 --text_col="pdf"
```

You should end up with file structure like this:

- /original:
  - /images
  - /files
  - /all_labels.csv

**Next, run ocr_migration.py.**

It takes two arguments:

- a config (which captures RANSAC parameters as well as the pair of folders that contain the old (with labels) and new ocr)
- the location where an excel summary should be saved.

This will produce the new folder which contains the migrated labels:

- /new:
  - /images
  - /files
  - /all_labels.csv

**Finally, push the labels to the platform with uploader.py.**

```

```
