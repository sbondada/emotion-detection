Emotion Detection
===
its the implementation of emotion detection methods for real time detection.though the present version of code has two methods implemented 
1. Fisher linear Discriminant based method
-----------------------------------------
2. point tracking based methods
----------------------------------


1.Fisher Linear Discriminant based method: this method involves gathering the data set ,we have used cohn-kanade dataset.we modified the dataset structure to so that it has three emothions 0-surprice,1-sad,2-smile.and calculated the fisher faces for each image and passed to the classifier.
to run this method you just need to use the emotion_detection.py with the image which you want to predict the emotion on and run the method

dependency:numpy,opencv,and set the pythonpath to facerec/py folder in the repo

the repo has other different codes for face tracking,corner detection,feature extractions and so on
