# Art Style Transfer
<em>Implementation and further improvement of the classical Artistic Neural Style Transfer algorithm by Gatys et al.</em>

<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/result-birds.png?raw=true" style="width: 600pt"></img>
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/result-cars.png?raw=true" style="width: 600pt"></img>
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/result-columns.png?raw=true" style="width: 600pt"></img>
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/result-girls.png?raw=true" style="width: 600pt"></img>
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/result-lions.png?raw=true" style="width: 600pt"></img>

## Algorithm Description

The <a href="https://arxiv.org/abs/1508.06576">original algorithm by Gatys et al.</a> is based on the idea of using a few intermediate layers of a pretrained neural 
net to get features from the so-called "content" and "style" images and harnessing the obtained feature maps for 
transferring the corresponding features from the style image to the content one.

However, the classical algorithm has several weaknesses. First, it works well only for images, which size is limited by 
the sizes of the feature maps used. For example, if we are using convolutional layers from VGG19, which was trained 
for images of resolution 224 X 224 X 3, we can't get a larger output image of a good quality with the 
classical Gatys' algorithm.  

Secondly, it is very sensitive to the content image's color contrast level. In the regions of low contrast 
the algorithm converges very slowly leaving spots of "style baldness", while in the places where color gradient leaps 
abruptly, it tends to noticeably diverge with some ugly distortions of the original content image. These effects can be clearly seen on the images of a bird below. 

<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/birds-Gatys.png?raw=true" style="width: 800pt"></img>

To overcome the above flaws the following improvements were implemented:
* <b>A pyramid loss algorithm</b> to allow for obtaining images with resolution up to 2K having non-square shape. 
* <b>A complex-structured noise</b> was added to the starting image at the beginning of the optimization procedure. This noise 
allows to increase the convergence rate in the "smooth" regions and stabilize it at the "sharp" edges reducing noise repercussions around them.
Also, different noise levels help to amplify the style features of different sizes at once.

<em>The final version shows fast convergence and good visual quality, which doesn't depend on the target resolution of the output image.</em>

A few examples of quality for three resolution levels (see the next section for details). The first column corresponds to a single pyramid level (256 pixels 
for the shortest dimension of the optimizing image); the second column - to 2 levels of pyramid (512 pixels); the third column - 
to 3 levels (1024 pixels):

<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/bird_diff_lvls.png?raw=true" ></img>

## Pyramid Loss Algorithm
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
   * 384 X 256 - Level 0 content image. The similar approach applies to the style image, as well.
4. At each optimization step the optimizing image is also resized to each level's dimensions. That makes the levels of the same dimensions for the optimizing image (4 in our case).
5. For each level of the content-style image pairs and the corresponding level of the optimizing image, 
the original Gatys' loss function values are calculated and summed together. <em>This sum becomes the total loss function and is then minimized during the optimization procedure.</em>   

With this approach all the scales of features are extracted from the input images - the large features are amplified on
the small image levels, while the small features are extracted on the large ones.

## Adding Some Noise
The modifications mentioned above work in a quite satisfactory manner for images that have more or less uniform contrast level. 
But there is a noticeable problem with the image of a bird at a clean sky (see the 4 images above).
This picture suffers from 2 diseases: "the sky baldness" and "the edge distortion". Both could be overcome by adding 
some artificially generated noise at the start of the optimization. But what kind of noise should be used?

This algorithm was implemented in a step-by-step manner, in a process of gradual improvement of the output image visual 
quality.

* First, a simple pixel-wide normally-distributed color noise was added to the input content image. This allowed to increase the local contrast level 
and get rid of the most "bald" spots at the places of the lowest contrast. The results of using this part of the algorithm can be seen on 
the following images:

<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/birds-norm-noise.png?raw=true" style="width: 800pt"></img>

* Second, a multi-level structure with different granularity of the noise for each level was added. On each of 
user-specified levels different sizes of noise spots are used, triggering style features of different scales. 
The results of using the different noise scales approach (16 and 128 spots for the shortest dimension of the content image, i.e. 
a large and a medium-sized noise, relatively), together with the corresponding noise maps, can be seen on the figures below.

   Large-sized noise (noise map):

   <img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/noise_mask_16.jpg?raw=true" style="width: 150pt"></img>

   Large-sized noise (result):

   <img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/birds-noise-16.png?raw=true" style="width: 800pt"></img>

   Medium-sized noise (noise map):

   <img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/noise_mask_128.jpg?raw=true" style="width: 150pt"></img>

   Medium-sized noise (result):

   <img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/birds-noise-128.png?raw=true" style="width: 800pt"></img>

   There is a possibility to use the noise of different scales at the same time. For instance, while obtaining the promo images 
shown at the top of the page, noise levels with 9, 18, and 36 spots for the shortest dimension of the content image were used, 
together with the pixel-wide noise. 

* Third, instead of using a random normally-distributed color noise, it was decided to make a noise map by randomly permuting the pixels of the input style 
image. This improvement took away the colors that are irrelative to the style.


* Finally, a dependency on the absolute value of the local gradient was added (the larger gradient value is - the less 
noise has to be added to this region, and vice versa), as well as gaussian-like envelopes for each noise level. The latter 
modification was needed to lower the level of noise at the central part of an image, allowing for obtaining mostly 
large-scale features from there, and thus visually distinguish the central foreground objects from the more detailed 
background. The final version of the images can be seen on the promo images at the top of this page.

## Installation Guide (tested on Debian 12)
To install please clone the repository to your working machine and follow these simple steps:
1. Create and activate a new python virtual environment inside the project folder. 
   Also, make sure that the required system packages are installed: 
   ```bash
   $ sudo apt install python3.11-venv libgl1-mesa-glx libglib2.0-0 python3-pip
   ```
   Then create the virtual environment itself: 
   ```bash
   $ cd ArtStyleTransfer
   $ python -m venv ./venv
   ```
   Activate the new virtual environment:
   ```bash
   $ . venv/bin/activate
   ```

2. Install all the packages, which are listed inside the files `requirements-base.txt` and `requirements-torch.txt`:
   ```bash
   $ pip install -r requirements-base.txt
   $ pip install -r requirements-torch.txt
   ```
   The project expects a GPU compatible with CUDA 12.1.

3. To run the web UI of the "lab", just execute the command:
   ```bash
   $ python lab.py
   ```
4. To run the Telegram bot, first create a file `token_DO_NOT_COMMIT.py` in the current directory 
(see subsection [Obtaining Telegram bot token](#obtaining-telegram-bot-token))
5. After the bot is created and the token is obtained and set, you can just run the bot backend. 
   ```bash
   $ python tlbot.py
   ```
   
### Obtaining Telegram bot token
To run the Telegram bot, first create a file `token_DO_NOT_COMMIT.py` in the current directory with the following content:
   ```
   TOKEN = "YOUR_BOT_TOKEN"
   ```
   Bot token `"YOUR_BOT_TOKEN"` can be obtained via https://t.me/BotFather . 
   It is assumed that in the process of obtaining the token you will also create your own bot. The instructions of `BotFatther` are self-explanatory. 

   
## Installation Guide (Dockerfile)   
   It is also possible to use Dockerfile from the project folder to create a Docker image, which allows for automatically 
running the Telegram bot. To do this, first create a file `token_DO_NOT_COMMIT.py` in the current project directory 
(see subsection [Obtaining Telegram bot token](#obtaining-telegram-bot-token)). Then build the Docker image with the command
   ```bash
   $ docker build -t ast .  
   ```
   and run it by
   ```bash
   $ docker run ast
   ```

## User Interface
Two variants of asynchronous UI are implemented for the current project:
* <b>lab.py</b>: a web-interface, mostly for testing purposes, showing a table of pre-configured images (you can see screenshots of it on this page). 
* <b>tlbot.py</b>: a Telegram bot, which allows to set up a task and obtain a style-transferred image in a simple and casual way.   

## Usage Guide
### The Bot
To use the bot, just send to it a pair of images <em>in one message</em>. The first image will be taken as a content image,
the second - as a style. The bot will start working, producing an intermediate result each 20% of the progress.

### The Lab
Run `python lab.py`, after a couple of seconds (plus a bit of time for downloading the neural net pretrained data if you run 
it for the first time) it will start producing images and reporting them. To see the report, open your 
browser at `http://<host>:8080`. The page doesn't update itself, refresh it manually.

The `lab.py` app is not interactive. All the configuration is done in the code itself (see `config.py` for the default settings).

# Original Images
## Content
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/original-content-images.png?raw=true" style="width: 600pt"></img>

## Style
<img src="https://github.com/irenemizus/ArtStyleTransfer/blob/master/img/original-style-images.png?raw=true" style="width: 600pt"></img>
