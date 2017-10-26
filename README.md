## Google Cloud Storage Reader with Tensorflow
Reads binary images from a Google Cloud Storage (GCS) bucket for use on Google ml-engine.  

  Repository Contents:  
    reading_data_tensorflow.ipynb: Demonstrational notebook on how to read data with tensorflow.
      This notebook reads MNIST data (binary image files) and returns batches of tensors and associated labels for training.
      Files can be read locally or from GCS bucket. 
      
    GCS-reader.py: Demonstrational script on how to read data with tensorflow.
      This script reads binary image files and returns shuffled batches of tensors and associated labels for training.
      
For more info about tensorflow's input pipeline, see my blog post. 
