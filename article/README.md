# Article Folder #

This folder is structured as follows:


```
|─── article
|     |─── plots
|     |    |─── <classifier_type (e.g. audio)>
|     |    |     |─── <name of the model used>
|     |    |     |     |─── <plots>
|     |─── results
|     |    |─── <confusion_matrices>
|     |    |─── <log_files>
└─── article.pdf
```

---

### Used Naming Scheme: ###

- plots: <date_of_script_execution>_<stage_of_execution>_<phase_of_execution>_<acc - accuracy / loss - loss>.png
- confusion_matrices: <audio_classifier>_<video_classifier>_cf_<batch_size>_<learning_rate>.png
- log_files: <audio_classifier>_<video_classifier>_<batch_size>_<learning_rate>_<date_of_script_execution>.txt