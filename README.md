This repo contains the three networks that consist the pipeline of our projects.


Sketch2image: implementations of the work SketchyGAN.

## We trained a model and upload the model as /ckpt_wgan/stage1/model.ckpt-344999 .

## Prerequisites

- Python 3, NumPy, SciPy, OpenCV 3
- Tensorflow(>=1.7.0)
- A recent NVIDIA GPU

## Preparations

The path to data files needs to be specified in `input_pipeline.py`. See below for detailed information on data files.
You need to download ["Inception-V4 model"](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz), unzip it and put the checkpoint under `inception_v4_model`.
The Sketchy Database can be found [here](http://sketchy.eye.gatech.edu/).
Run the script /data_processing/converter.py to process the downloaded data into proper format.
Use ‘extract_images.py' under ‘data_processing’ to extract images from tfrecord files. You need to specify input and output paths. The extracted images will be sorted by class names.

## Configurations

To test the model, change ‘mode’ from ‘train’ to ‘test’ and fill in ‘resume_from’ in ‘main_single.py’.


Image2detection: contains the code and trained model from Faster-RCNN.

## Prerequisites

install tensorflow
	conda install -c anaconda tensorflow-gpu
some essential lib
	sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
	pip install --user Cython
	pip install --user contextlib2
	pip install --user jupyter
	pip install --user matplotlib
install protobuf
	cd research/
	protoc object_detection/protos/*.proto --python_out=.
add library to pythonpath

	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
a simple test all things well
	python object_detection/builders/model_builder_test.py

## run faster rcnn api

	cd object_detection/
	python object_detection_api.py /test/image/path

	e.g.: python object_detection_api.py ./n02691156_9491.jpg

## obtain the result

	demo.png


Img2poem: this folder contains the code of poem generation from images

## test images are under the folder images/

## pre-trained models are in model/, including image feature extraction model (object.params, scene.params, Sentiment.params) and poem generation model (ckpt/).

## code for testing are in src/
	to run the code
	test.py

	To test how much time it cost to generate a poem for an image

	def get_poem(image_file):
    	"""Generate a poem from the image whose filename is `image_file`

   	 Parameters
    	----------
    	image_file : str
    	Path to the input image

    	Returns
    	-------
    	str
        	Generated Poem
    	"""