# MP6: Histogram

The purpose of this lab is to implement an efficient histogramming equalization algorithm for an input image. Like the image convolution MP, the image is represented as RGB float values. You will convert that to GrayScale unsigned char values and compute the histogram. Based on the histogram, you will compute a histogram equalization function which you will then apply to the original image to get the color corrected image.

## The code performs the following:
* Cast the image to unsigned char

* Convert the image from RGB to Gray Scale. You will find one of the lectures and textbook chapters helpful.

* Compute the histogram of the image

* Compute the scan (prefix sum) of the histogram to arrive at the histogram equalization function

* Apply the equalization function to the input image to get the color corrected image

## Authors

Jiangyan Feng - jf8@illinois.edu

Ph.D. Candidate

Chemical and Bimolecular Engineering

University of Illinois at Urbana-Champaign

200 RAL, 600 S. Mathews Ave., Urbana IL 61801
