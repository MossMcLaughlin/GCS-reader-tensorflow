import tensorflow as tf 


def read_APS(filename_queue,image_shape):
    """ Reads aps file format (easily changed for other binary image files),
	  from local destinations or GCS bucket. 

	Args: filename queue: Tensorflow queue of files to read.
              image_shape: List containing size of image 
                  dimensions, [height,width,depth].

	Returns: result. Contains attributes:
	    uint8_image: Image data of type.uint8
	    label: Associated training label  
            key: Name of file read 
    """
    class APSrecord(object):
        pass
    result = APSrecord()

    result.height = image_shape[0]
    result.width = image_shape[1]
    result.depth = image_shape[2]
    image_bytes = result.height * result.width * result.depth
    label_bytes = 512 
    record_bytes = image_bytes + label_bytes
    
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    print("Built reader")
    result.key, value = reader.read(filename_queue)
    print("read")
    # Convert from a string to a vector of uint16 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint16->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    print("sliced label")
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    result.uint8_image = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.height, result.width, result.depth])
    # Convert from [depth, height, width] to [height, width, depth].
    print("sliced data")

    return result

def _generate_image_and_label_batch(image, label,
                                    batch_size, shuffle,min_queue_examples=1):
    """Construct a queued batch of images and labels.
    Args:
      image: rank 3 Tensor of [height, width, depth] of type.float32.
      label: rank 1 Tensor of type.int32.
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. rank 4 tensor of [batch_size, height, width, depth] size.
      labels: Labels. rank 2 tensor of [batch_size,num_classes] size,
          where num_classes is the length of our models output vector. 
    """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    print("entering visualizer")
    #tf.summary.image('images', images)
    print("exiting visualizer")
    return images, tf.reshape(label_batch, [batch_size])

def gen_inputs(file_list,IMAGE_SHAPE):
    """ Generates queued, batched training data and labels from
            the list of files we want to read. 

	Args: 
      	  file_list: list of files to read.
      	  IMAGE_SHAPE: shape of BATCHED images, 
                [batch_size, height, width, depth].
		
	Returns:
      	  images: Images. rank 4 tensor of [batch_size, height, width, depth] size.
          labels: Labels. rank 2 tensor of [batch_size,num_classes] size,
            where num_classes is the length of our models output vector.
    """
    for f in file_list:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        else: print("Found file: " + f)

    filename_queue = tf.train.string_input_producer(file_list)
    
    read_input = read_APS(filename_queue,IMAGE_SHAPE[1:])

    reshaped_image = tf.cast(read_input.uint8_image, tf.float32)
    
    ###   Simple Image pre-processing   ###

    
    batch_size = IMAGE_SHAPE[0]
    
    # Change these if we want to crop or pad.
    height = IMAGE_SHAPE[1]
    width = IMAGE_SHAPE[2]
    depth = IMAGE_SHAPE[3] # Depth is the same as num_channels
    
    # Here we can crop the image if desired.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors. (shape must be definied before we batch)
    float_image.set_shape([height, width, depth])
    read_input.label.set_shape([1])

    return _generate_image_and_label_batch(float_image, read_input.label, batch_size,shuffle=False)
    
