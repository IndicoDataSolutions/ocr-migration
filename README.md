# ocr-migration
### Scripts for Migrating Labeled Data from one OCR engine to another

**First, run [get_datasets.py](https://github.com/IndicoDataSolutions/ocr-migration/blob/main/get_datasets.py) to gather OCR + Labels for the original dataset and OCR for the new dataset.**

You should end up with file structure like this:
- /original:
   - /images
   - /files
   - /all_labels.csv
- /new:
   - /images
   - /files
   - /all_labels.csv
   
**Next, run ocr_migration.py.**

It takes two arguments: 
   - a config (which captures RANSAC parameters as well as the pair of folders that contain the old (with labels) and new ocr)
   - the location where an excel summary should be saved.
   
**Finally, push the labels to the platform with uploader.py.**
