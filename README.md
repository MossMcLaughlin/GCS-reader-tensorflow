## Google Cloud Storage Reader with Tensorflow  
Reads binary images from a Google Cloud Storage (GCS) bucket for use on Google ml-engine.  

Repository Contents:  

<em>reading_data_tensorflow.ipynb</em>: Demonstrational notebook on how to read data with tensorflow.
      This notebook reads MNIST data (binary image files) and returns batches of tensors and associated labels for training.
      Files can be read locally or from GCS bucket.    

<em>GCS-reader.py</em>: Demonstrational script on how to read data with tensorflow.
      This script reads binary image files and returns shuffled batches of tensors and associated labels for training.
      
For more info about tensorflow's input pipeline, see my blog post. 
