# April-Tag-Detection-and-Tacking
All the plots are saved in /results folder. While running the files, exit any of the plots by pressing the 'q' key
Test video found [here](https://drive.google.com/file/d/1MWJOLJcFJvRporEfZ-j_lPbGkVuRv0So/view)

## Problem 1 
a) This will give the output of FFT on the first frame of the test video
b) This will also give the tag IDs of the two AR tags in the data/ folder, ar_tag.png and test.jpg

    python3 get_edges.py

## Problem 2
### 2a. Superimpose Testudo image
This file will superimpose the Testudo image onto the AR tag in the video

	 python3 superimposition.py
A video of this can be found [here](https://drive.google.com/file/d/1VplSTPKKCVewUl7ahmtW_9ra3Wc85yw8/view?usp=sharing)

### 2b. 3D Cube Projection
This file will project a 3D cube onto the AR tag in the video 

    python3 augmented_reality.py
A video of this can be found [here](https://drive.google.com/file/d/1wJ_TR2KZmbtDgHyG_QL4jqPfEcMAVapM/view?usp=sharing)

