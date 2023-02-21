# ocr-migration
Scripts for Migrating Labeled Data from one OCR engine to another

First run [get_datasets.py](https://github.com/IndicoDataSolutions/project-fruitfly/blob/main/dataset_scripts/get_datasets.py) to gather OCR + Labels for the original dataset and OCR for the new dataset.

You should end up with file structure like this:
original
   |/images
   |/files
   |all_labels.csv
new
   |/images
   |/files
   |all_labels.csv
   
