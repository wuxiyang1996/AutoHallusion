Both files are used to pre-process the MSCOCO validation dataset for our pipeline. Files need to be placed under the dataset folder in the following structure, while `args_cmd.dataset_dir` is `datasets`, `dataset_raw_data` is `coco_val_sets`, `dataset_obj_db` is `obj_db_clean.json` or `corr_obj_db_clean` (only for **Paired Object Insertion**) and `dataset_scene_db` is `scene_db_clean.json`:
``` 
├── datasets
│      └── coco_val_sets
|      └── obj_db_clean.json
|      └── scene_db_clean.json
|      └── corr_obj_db_clean.json
```
