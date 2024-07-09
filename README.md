# Art Style Transfer
<em>Implementation and further improvement of the classical Artistic Neural Style Transfer algorithm by Gatys et al.</em>

The <a href="https://arxiv.org/abs/1508.06576">original algorithm by Gatys et al.</a> is based on the idea of using a few intermediate layers of a pretrained neural 
net to get features from the so-called "content" and "style" images and harnessing the obtained feature maps for 
transferring the corresponding features from the style image to the content one.

However, the classical algorithm has several weaknesses. First, it works only for images, which size is limited by 
the sizes of the feature maps used. For example, if we are using convolutional layers from VGG19, which was trained 
for images of resolution 224 X 224 X 3, we can't get a larger output image of a good quality with the 
classical Gatys's algorithm.  

Secondly, it is very sensitive to the content image's color contrast level. In the regions of low contrast 
the algorithm converges very slowly leaving spots of "style baldness", while in the places where color gradient leaps 
abruptly, it tends to noticeably diverge with some ugly distortions of the original content image. The examples of applying 
the classical Gatys et al. algorithm to several pairs of content-style images are shown on the Fig. 1. 

To overcome the above flaws the following improvements were implemented:
* <b>A pyramid loss algorithm</b> to allow for obtaining images with resolution up to 2K having non-square shape. 
* <b>A complex-structured noise</b> was added to the starting image at the beginning of the optimization procedure. This noise 
allows to increase the convergence rate in the "smooth" regions and stabilize it at the "sharp" edges reducing noise repercussions around them.
Also different noise levels help to amplify the style features of different sizes at once.

## Pyramid loss algorithm
This algorithm implements an idea of minimizing loss function values for several sizes of the content-style pairs of images
<em>simultaneously</em>. To achieve that the following steps are performed:
1. Aligning the original content and style images by the smaller dimension to a chosen power of 2 by resizing them bicubically.
2. The chosen size is calculated according to a simple formula dim = 256 * 2^L, where L is an integer parameter called "the levels number". 
Let's assume (for example) that L=3. In this case dim=2048. So if the original content image has an aspect ratio of 3/2, 
its dimensions become 3072 X 2048.
3. Before starting the optimization, a set of "pyramid levels" - smaller images with binary-reduced resolutions - is prepared
by resizing the content and style images. 
So we have 4 levels of the following sizes:
   * 3072 X 2048 - Level 3 content image (the original one),
   * 1536 X 1024 - Level 2 content image,
   * 768 X 512 - Level 1 content image,
   * 384 X 256 - Level 0 content image. The similar approach applies to the style image.
4. At each optimization step the optimizing image is also resized to the each level's dimensions. That makes the levels of the same dimensions for the optimizing image (4 in our case).
5. For each level of the content-style image pairs and the corresponding level of the optimizing image, 
the original Gatys' loss function values are calculated and summed together. <em>This sum becomes the total loss function and is then minimized during the optimizing procedure.</em>   

With this approach all the scales of features are extracted from the input images - the large features are amplified on
the small image levels, while the small features are extracted on the large ones.

## Noise adding
All the mentioned above results in a quite satisfactory quality for images that have more or less uniform contrast 
(like the picture of a car or columns). But there is a noticeable problem with the image of a bird at a clean sky.
This picture suffers from 2 diseases: "the sky baldness" and "the edge distortion". Both could be overcome by adding 
some artificially generated noise at the start of the optimization. But what kind of noise should be used?

This algorithm was implemented in a step-by-step manner, in a process of gradual improvement of the output image visual 
quality.

* First, a simple pixel-wide normally-distributed color noise was added to the input content image. This allowed to increase the local contrast level 
and get rid of the most "bald" spots at the places of the lowest contrast. The results of using this part of the algorithm can be seen on 
the Fig. 2.  
* Second, a multi-level structure with different granularity of the noise for each level was added. On each of 
user-specified levels different sizes of noise spots are used, triggering style features of different scales. 
The results of using the different noise scales approach (16 and 128 spots for the shortest dimension of the content image, i.e. 
a large and a medium-sized noise, relatively), together with the corresponding noise maps, can be seen on the figures below.
* Third, instead of using a random normally-distributed color noise, it was decided to make a noise map by randomly permuting the pixels of the input style 
image. This improvement took away the colors that are irrelative to the style. This stage of the algorithm is illustrated on Fig. 5. 
* Finally, a dependency on the absolute value of the local gradient was added (the larger gradient value is - the less 
noise has to be added to this region, and vice versa), as well as gaussian-like envelopes for each noise level. The latter 
modification was needed to lower the level of noise at the central part of an image, allowing for obtaining mostly 
large-scale features from there, and thus visually distinguish the central foreground objects from the more detailed 
background. The final version of the images can be seen on Fig. 6.

## User interface
Two variants of asynchronous UI are implemented for the current project:
* <b>lab.py</b>: a web-interface, mostly for testing purposes, showing a table of pre-configured images (you can see screenshots of it on this page). 
* <b>tlbot.py</b>: a Telegram bot, which allows to set up a task and obtain a style-transferred image in a simple and casual way.   

