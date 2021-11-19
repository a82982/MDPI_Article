# Deep Learning for Activity Recognition - MDPI Article #

## This repository serves to make available: ##

1. The data discovered from the article that prove the achieved results;

2. The code used in the article;

3. The dataset used in the article;

## File / Folder Structure: ##

- [article](./article/) - contains the data generated during the conducted study, for more detailed information, navigate to the folder and check its documentation;
- [classes](./classes/) - contains the auxiliary classes, such as the definition of the __Early Fusion Model__, that facilitate the running of the program;
- [data](./data/) - contains the dataset that was used in order to train and validate the training of the different models;
- [models](./models/) - created when executing the script, it contains the models that are trained and validated during the execution of the script;
- [plots](./plots/) - created when executing the script, it contains the plots that are drawn during the execution script, in order to evaluate the models' performance;
- [results](./results/) - created when executing the script, it contains additional files that allow the evaluation of the models performance, such as __Confusion Matrices__ and a file that logs the __Accuracy__ and __Loss__ values for each of the different models on each epoch and phase;
- [3dmain.py](./3dmain.py) - script that is used to create, train, and evaluate the models;

## How to Run the Code: ##

1. Create a __Python__ virtual environment: `virtualenv <name>`
2. Activate the virtual environment: `source <name>/bin/activate`
3. Install the dependencies from __requirements.txt__: `pip install -r requirements.txt`
4. Change the following values according to your training preferences:
    - *num_of_videos* ([violent_dataset.py](./classes/violent_dataset.py) 's line 95): to change the number of videos that will be used - __MUST BE THE SAME AS num_videos MENTIONED BELOW__
    - *audio_model* (3dmain.py 's line 67): to change the *audio* classifier that will be used
    - *video_model* (3dmain.py 's line 69): to change the *video* classifier that will be used
    - *learn_rate* (3dmain.py 's line 73): to change the classifiers' learning rate
    - *num_epochs* (3dmain.py 's line 75): to change the number of epochs that will be used
    - *num_videos* (3dmain.py 's line 77): to change the number of videos that are being evaluated - __MUST BE THE SAME AS num_of_videos MENTIONED ABOVE__
    - *b_size* (3dmain.py 's line 79): to change the batch size that will be used to train the models
5. Execute the script: `python 3dmain.py`