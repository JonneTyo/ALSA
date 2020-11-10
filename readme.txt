For both CrackTrain and CrackMain:
-	Extract a .png image of the area to be analyzed
	-	This image should have black background
	- 	If this image is used for training, the quality of the image should be same across the images
	-	If this image is used for prediction, the quality of the image should be around the same as used for the training
	-	This image needs to be the smallest rectangle that covers the area
	-	THE NAME OF THIS .PNG IMAGE MUST BE A SUBSTRING OF THE SHAPEFILES
		-	If the name of the .png image is 'ABC123.png', the shapefiles must have 'ABC123' in their filenames somewhere.
		-	For this reason, if you have shapefiles named 'abc_1.shp' and 'abc_2.shp', don't name the .png image as 'abc.png' as it can confuse the 2 shapefiles.
-	Install the packages described in the requirements.txt


For prediction:

-	The program first asks for the .png image's relative or full path (including the .png at the end). Type it in.
-	The program then asks for the path to the .shp-file containing the polygon of the area to be analyzed.
-	The program then asks for the path to the .hdf5-file containing the weights of the CNN-model. By default, this is named 'unet-weights.hdf5'. If not found, try to train model first.
-	Finally the program asks for the name of the .shp-file to be produced.

For training:

-	The CrackTrain looks for Training folder and contents within it. If this is missing, run the module once and it creates them.
-	Navigate to Training\Shapefiles
	-	\Areas should contain the .shp files containing the polygon of the area to be analyzed.
	-	\Labels should contain the .shp files containing the lines you wish the program detects.
-	Navigate to Training\Images\Originals
	-	Place the .png images you wish to train for in here.
-	THE FOLDER Training\Images\Generated IS CLEARED AT THE START OF THE PROGRAM! DO NOT STORE ANYTHING HERE!
-	Running the CrackTrain module will create/overwrite a file named 'unet_weights.hdf5'. This is the file that's to be used when predicting.
