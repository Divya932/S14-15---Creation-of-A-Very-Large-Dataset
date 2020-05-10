# S14-15: Creation of A Very Large Dataset

This Repo explains how you can, not just use available datasets in your journey of DNN but efficiently create them yourself.

## Description  
We have used approximately 100 background images and 100 foreground objects to create a massive dataset with more than 400,000 images.  
Also, corresponding masks for each resultant as well as foreground objects have been created.  
We ultimately create depth maps of all images in our dataset.  
A special focus has been laid on handling such massive sized data efficiently with limited resources.

## Starting with..
### Background  
Background contains pictures of empty roads of all sorts, ranging from city roads to flyovers.   
[Here](https://drive.google.com/open?id=1CD2lC-16tWiOmL7dm73nNr_XfoFQL10v) is a link to all our background images.  
  
<b>Statistics</b>  
* Image dimensions: 224\*224 pixels  
* Image format: jpg - We take a special care to avoid png format here in order to save space.  
* Number of Images: 115  
* Folder size: 1.06MB
* Background mean: [0.4056645395930562, 0.40100307446352884, 0.39399309587970394]
* Background std: [0.25793691468858726, 0.2594673166338195, 0.27366606302821606]

<p float="left">
  <img src="/background/S_50.jpg" width="150" />
  <img src="/background/S_3.jpg" width="150" /> 
  <img src="/background/S_73.jpg" width="150" />
  <img src="/background/S_102.jpg" width="150" />
  <img src="/background/S_57.jpg" width="150" />
</p>

### Foreground  
Foreground contains object as car on a transparent background in all possible orientations.  
The background can be made transparent using multiple tools like GIMP, Paint 3D or Microsoft PowerPoint.  
In our work, we have used Paint 3D to acheive the transparency and later saved it to .png format to preserve the alpha channel.  
[Here](https://drive.google.com/open?id=1J_yMHZ_cP93ZfL197KuxilTiXTrANsup) is a link to our foreground objects  
  
<b>Statistics</b>  
* Image dimensions: varies from 94px to 120px (aspect ratio maintained to fit background)
* No.of channels: 4
* Image format: png - preserves the alpha channel(transparency)
* Number of Images: 200 (100 non-flipped + 100 horizontal flipped)
* Folder Size: 2.17MB
* Foreground mean: [0.2094951758817645, 0.19539478564017493, 0.19461715970327925]
* Foreground std: [0.2836906317489486, 0.2668295825763676, 0.26662684995218194]

<p float="left">
  <img src="/foreground/Car_0.png" width="100" />
  <img src="/foreground/Car_73.png" width="100" /> 
  <img src="/foreground/Car_23.png" width="100" />
  <img src="/foreground/Car_186.png" width="100" />
  <img src="/foreground/Car_75.png" width="100" />
  <img src="/foreground/Car_19.png" width="100" />
  <img src="/foreground/Car_70.png" width="100" />
  <img src="/foreground/Car_83.png" width="100" />
</p>  

### Foreground Mask
Foreground masks have been created using the concept of alpha channel in the png images.  
* The Alpha Channel - controls the transparency factor in our images. It ranges from 0(completely transparent) to 255(fully opaque).  
* We make use of this channel to create our fg masks by filling in binary values, 0 for all the pixels where alpha = 0 and 255 for alpha = non-zero.  
This gives us our corresponding masks.
All our foreground masks can be found [here](https://drive.google.com/open?id=1w-fjH1UeutYIW7lNcCWBD3BK1NZ6ytxg)  
  
<b>Statistics</b>
* Image format: jpg
* Number of channels: 1
* Number of Images: 200
* Folder size: 400KB  
* Fg_Mask Mean : 0.4901594366127988
* Fg_Mask Std: 0.48846756739825214

<p float="left">
  <img src="/fg_mask/Car_0_mask.jpg" width="100" />
  <img src="/fg_mask/Car_73_mask.jpg" width="100" />
  <img src="/fg_mask/Car_23_mask.jpg" width="100" />
  <img src="/fg_mask/Car_186_mask.jpg" width="100" />
  <img src="/fg_mask/Car_75_mask.jpg" width="100" />
  <img src="/fg_mask/Car_19_mask.jpg" width="100" />
  <img src="/fg_mask/Car_70_mask.jpg" width="100" />
  <img src="/fg_mask/Car_83_mask.jpg" width="100" />
</p>  

### Fg_Bg Together: OverLayed Images  
Foreground objects are overlayed on each background image at 20 random positions.  
This gives a total of 115\*200\*20 = 4,60,000 images.  
We choose a random location each time we overlay an fg object on bg from a list of random positions. PIL's Image.paste is used for the purpose.  
Find the dataset [here](https://drive.google.com/drive/folders/1Ul4mQ9BO41sjsV0HNaQlCk0wznMEBH9X?usp=sharing) (Divided into batches of 1000 images each).  
  
<b>Statistics</b>  
* Image dimensions: 224\*224\*3
* No.of channels: 3
* Image format: jpg - saves space as we don't need the transparency(alpha) channel here.
* Number of Images: 4,60,000
* Folder Size: 4.96GB
* Dataset Mean: [0.40456055804985874, 0.3983824357134671, 0.3925343274986984]
* Dataset Std: [0.2599649396912192, 0.2609780929820059, 0.27395228266780175]


Here's a peek into few of our images:  

<p float="left">
  <img src="/fg_bg/P_1.jpg" width="150" />
  <img src="/fg_bg/P_206074.jpg" width="150" />
  <img src="/fg_bg/P_100071.jpg" width="150" />
  <img src="/fg_bg/P_148576.jpg" width="150" />
  <img src="/fg_bg/P_156303.jpg" width="150" />
</p>

### Fg_Bg Masks  
These masks are created by overlaying the foreground masks on a black patch with size equal to background image's size.  
* To save space we take an important step - We create a single channel in our masks by creating a black patch of size equal to bg's size(224\*224)  out of a simple 2D array. We then paste our fg masks at the same location we overlayed our fg object.
Find the entire Masks zip file [here](https://drive.google.com/open?id=1pzptuqtqqqSvQekhFEgCkQN9Neay-Yim) 
  
<b>Statistics</b>  
* Image dimensions: 224\*224\*1
* No.of channels: 1
* Image format: jpg - saves space as we don't need the transparency(alpha) channel here.
* Number of Images: 4,60,000
* Folder Size: 723MB
* Dataset Mask Mean: 0.07221486453923726
* Dataset Mask Std: 0.2556922185275919 

The corresponding masks are:  

<p float="left">
  <img src="/fg_bg_mask/mask_1.jpg" width="150" />
  <img src="/fg_bg_mask/mask_206074.jpg" width="150" />
  <img src="/fg_bg_mask/mask_100071.jpg" width="150" />
  <img src="/fg_bg_mask/mask_148576.jpg" width="150" />
  <img src="/fg_bg_mask/mask_156303.jpg" width="150" />
</p>

### Fg_Bg Depth Maps  
Our Monocular Depth Estimation Maps have been produced using a pre-trained DenseNet-201 model, cloned from [this](https://github.com/ialhashim/DenseDepth) repo with minor modifications. Our modifications mainly include scaling up of each image during processing to get a better view and hence a better prediction of depth for each object. The images are loaded at size 224x224, scaled up to 640x480 during processing, and then scaled down back to 224x224 before saving.  
Lack of a Depth camera or a LIDAR camera leads us to rely on pretrained DenseNet-201 model to create depth maps.   
Clone this [Depth Model](https://github.com/ialhashim/DenseDepth) to generate your own depth maps.   
Find our modified version of certain files [here](https://github.com/Divya932/S14-15---Creation-of-A-Very-Large-Dataset/tree/master/Modified_files)  
Find a link to the entire depth map zip file [here](https://drive.google.com/open?id=1cNlUKFKTHbAlpU_KdyIDUKgHDigjbJBs)  (976.6MB)
  
<b>Statistics</b>  
* Image dimensions: 224\*224\*1
* No.of channels: 1
* Image format: jpg - saves space as we don't need the transparency(alpha) channel here.
* Number of Images: 400,000
* Zip Folder Size: 976.6MB
* Depth Map Mean: 0.3955709002351957
* Depth Map Std: 0.26238287425746376  

Here's a glimpse of the depth maps generated:

<p float="left">
  <img src="/Depth maps/depth_P_1.jpg" width="150" />
  <img src="/Depth maps/depth_206074.jpg" width="150" />
  <img src="/Depth maps/depth_P_100071.jpg" width="150" />
  <img src="/Depth maps/depth_100792.jpg" width="150" />
  <img src="/Depth maps/depth_156303.jpg" width="150" />
</p>  
  
Find the Data Preprocessing and Statistics code [here](https://github.com/Divya932/S14-15---Creation-of-A-Very-Large-Dataset/blob/master/Ass-15A-%20Preprocessing%20and%20stats.ipynb)

