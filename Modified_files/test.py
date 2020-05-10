import os
import glob
import argparse
import matplotlib
from PIL import Image
import numpy as np
import io
import glob


# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--batch', default='0', type = int, help='Batch Number')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
ex, viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()

filearr = []
c =0
src_path = '/content/DenseDepth/Deep Learning/MaskRCNN/bg_fg/'
for cn, filenm in enumerate(os.listdir(src_path)):
    filearr.append(filenm)

print("No. of files to process: ", len(filearr))

for i in range(len(filearr)):
    im = Image.fromarray(np.uint8(ex[i]*255)).resize((224,224))
    im.save('/content/DenseDepth/Depth_Masks/depth_' + filearr[i])
    c += 1
    os.remove(os.path.join(src_path,filearr[i]))
print(c," Files saved")