from PIL import Image
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import keras
from keras.preprocessing import image
import numpy as np
import os
from keras.models import Sequential, Model, model_from_json
def load_keras_model(model_path):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def get_model():
    vgg_model = keras.applications.VGG16(include_top=True, weights='imagenet')
    vgg_model.layers.pop()
    vgg_model.layers.pop()
    inp = vgg_model.input
    out = vgg_model.layers[-1].output
    model = Model(inp, out)
    return model

def get_features(model, cropped_image):
    x = image.img_to_array(cropped_image)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    features = model.predict(x)
    return features

WORD2VECPATH    = "../data/class_vectors.npy"
MODELPATH       = "../model/"

def main():

    IMAGEPATH = 'Human-Hands-Front-Back.jpg'
    img         = Image.open(IMAGEPATH).resize((224, 224))
    vgg_model   = get_model()
    img_feature = get_features(vgg_model, img)
    img_feature = normalize(img_feature, norm='l2')
    model       = load_keras_model(model_path=MODELPATH)
    pred        = model.predict(img_feature)
    class_vectors       = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames          = list(classnames)
    vectors             = np.asarray(vectors, dtype=np.float)
    tree                = KDTree(vectors)
    dist, index         = tree.query(pred, k=5)
    pred_labels         = [classnames[idx] for idx in index[0]]
    print()
    for i, classname in enumerate(pred_labels):
        print("%d- %s" %(i+1, classname))
    print()
    return

if __name__ == '__main__':
    main()