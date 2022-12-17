# install mtcnn==0.1.0,tensorflow==2.3.1, keras==2.4.3, keras-vggface==0.6, keras_applications==1.0.8

# import os
# import pickle
# actors = os.listdir('data')
# print(actors)
# filenames=[]
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filenames.append(os.path.join('data',actor,file))
#
# print(len(filenames))
# pickle.dump(filenames,open('filenames.pkl','wb'))

# Omkar Parkhi in the 2015 paper titled “Deep Face Recognition.”

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# print(model.summary())

def function_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result=model.predict(preprocessed_img).flatten()
    return result

features=[]
for file in tqdm(filenames):
    features.append(function_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))

