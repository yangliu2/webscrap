import numpy as np
import json
import h5py
import os.path
import argparse
import shutil
import math



parser = argparse.ArgumentParser(description='Train a multicategory classification model for categories of images found in the source folder')
parser.add_argument('-s','--source', help='Source directory where images can be found, grouped into category folders. The default is the current path.', required=False, default='./')
parser.add_argument('-d','--destination', help='Destination directory where model will be located The default is [CURRENT PATH]/model', required=False, default='./model')
parser.add_argument('-p','--pctval', help='Percentage of images to use for validation. The default is 10.', required=False, default='10')
parser.add_argument('-e','--epochs', help='The number of epochs to train for The default is 50.', required=False, default='50')
parser.add_argument('-b','--batch_size', help='The number of images to run in a single batch.', required=False, default='16')
parser.add_argument('-sw','--scale_width', help='The width to scale the input images to. The default is 150.', required=False, default='299')
parser.add_argument('-sh','--scale_height', help='The height to scale the input images to. The default is 150.', required=False, default='299')
parser.add_argument('-mt','--model_type',  help='The type of base model to use for transfer learning. Acceptable values are vgg16, inception and resnet50', required=False, default='vgg16')
parser.add_argument('-rr','--rotation_range', help='Randomly rotates the image within the given range prior to processing', required=False, default='0')
parser.add_argument('-mi','--max_iterations_over_sample_directories', help='The maximum number of complete passes over the folder of images to generate samples', required=False, default='1')
parser.add_argument('-hf','--horizontal_flip', action='store_true', help='Horizontally flips the images randomly prior to processing')
parser.add_argument('-vf','--vertical_flip', action='store_true', help='Vertically flips the images randomly prior to processing')
parser.add_argument('-sd','--random_seed', help='A random long int used to seed the random number generator', required=False, default='12345')
parser.add_argument('-sc','--save_checkpoints', help='Saves the model checkpoints with each epoch', action='store_true')
parser.add_argument('-sb','--save_best_model', help='Saves only the best model in the checkpoints', action='store_true')
parser.add_argument('-cw','--use_class_weight', help='Uses Class Weight in an attempt to adjust a loss function when there in an imbalance between the classes', action='store_true')
parser.add_argument('-lr','--learning_rate', help='The learning rate for the optimizer of the model', required=False, default='.0001')
parser.add_argument('-o','--optimizer', help='The optimizer to use for the top model. Valid values are ADAM, RMSPROP and SGD', required=False, default='SGD')


args = parser.parse_args()

np.random.seed(long(args.random_seed))

# dimensions of our images.
img_width = int(args.scale_width)
img_height = int(args.scale_height)
destination_dir = args.destination
full_model_path = os.path.join(destination_dir,'multi_category_classifier_model.h5')
source_path = args.source
train_data_dir = os.path.join(source_path,'train')
validation_data_dir = os.path.join(source_path,'validation')
epochs = int(args.epochs)
model_type = args.model_type
preferred_size = (0,0)

pctval = int(args.pctval)

batch_size = int(args.batch_size)
max_iterations_over_sample_directories = int(args.max_iterations_over_sample_directories)
rotation_range = int(args.rotation_range)
horizontal_flip = args.horizontal_flip
vertical_flip = args.vertical_flip

# keras imports occur futher down so that tensorflow libraries are not loaded until all of the arguments have been parsed.
from rlcpmodelcheckpoint import RLCPModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras import applications, optimizers
from keras.callbacks import ModelCheckpoint
from keras.engine.training import Model
from keras.models import load_model




def main():
    base_model = create_base_model()
    features = generate_train_and_validation_features(base_model)
    train_and_save_full_model(base_model, features)
    #test_saved_model(features)


def load_model_and_metadata():
    model = load_model(full_model_path)

    # Load the H5 file    
    f = h5py.File(full_model_path, mode='r') # Open file in read (r) mode
    metadata = json.loads(f.attrs['metadata'].decode('utf-8'))    

    return (model, metadata)


def test_saved_model(features):
    print("Testing predictions by loading the model that was saved to disk...")
    model, metadata = load_model_and_metadata()
    test_predictions(model, features["validation_class_data"])


def get_label_array(datagen, dataDir):

    cat_generator = datagen.flow_from_directory(
        dataDir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    #cat_generator.samples is the total # of files across all sub directories (classifications)
    max_samples = cat_generator.samples * max_iterations_over_sample_directories
    print('Max Iteration: '+str(max_iterations_over_sample_directories)+', Samples: '+str(max_samples))

    labels = None
    print("Generating Labels for "+dataDir)
        
    batch_count = 0
    for image_batch, label_batch in enumerate(cat_generator):
        batch_count+=1

        #if there are any remaining under the batch size then stop
        #or if we have processed ALL of the images then stop
        #NOTE: The generator will keep going on and on infinetly by design if we don't do this.

        if(len(label_batch[1])<batch_size or (batch_count * batch_size) > max_samples):
            break

        if(labels == None):
            labels = np.zeros(shape=(batch_size, len(label_batch[1][0])))
        else:
            labels = np.resize(labels,((batch_count*batch_size), len(label_batch[1][0])))
        
        starting_index = ((batch_count-1) * batch_size)

        for i in range(batch_size):
            one_row = label_batch[1][i]
            #print("batch_count, Index: "+str(batch_count)+","+str(starting_index+i))
            labels[starting_index+i] = one_row

    # print("Labels for "+dataDir)
    # print(labels)
    return {
        "class_map": labels,
        "classes": getClasses(cat_generator)
    }

#Make sure to sort the classes.  Keras trains based on alphabetical order of the sub directories for classification
def getClasses(cat_generator):
     return sorted(list(cat_generator.class_indices.keys()))

def get_datagen():
    return ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotation_range=rotation_range)

def generate_bottleneck_features(class_map, dir_type, data_dir, datagen, base_model):
    total_batches = len(class_map) // batch_size
    print(dir_type+" Images Size : " + str(total_batches))

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features = base_model.predict_generator(
        generator, total_batches, verbose=1)
    #np.save(open('bottleneck_features_train.npy', 'w'),
    #        bottleneck_features_train)
    return bottleneck_features

def generate_train_and_validation_features(base_model):

    train_class_data = get_label_array(get_datagen(), train_data_dir)
    train_bottleneck_features = generate_bottleneck_features(train_class_data["class_map"],"train",train_data_dir, get_datagen(), base_model)

    validation_class_data = get_label_array(get_datagen(), validation_data_dir)
    validation_bottleneck_features = generate_bottleneck_features(validation_class_data["class_map"], "validation", validation_data_dir, get_datagen(), base_model)

    return {
        "train_class_data":train_class_data,
        "validation_class_data": validation_class_data,
        "train_bottleneck_features": train_bottleneck_features,
        "validation_bottleneck_features": validation_bottleneck_features
    }



def create_base_model():
    print('Attempting to get base model: '+model_type)
    base_model = get_base_model(model_type)
    return base_model

def preferred_size_matches_scale():
    return ((preferred_size[0] == img_width) and (preferred_size[1] == img_height))

def get_base_model(model_type):
    input_tensor = Input(shape=(img_width,img_height,3))
    input_shape = shape=(img_width,img_height,3)
    if(model_type == 'vgg16'):       
        # build the VGG16 network
        #NOTE: VGG16 code needs the base model downloaded or else it will try to download it
        preferred_size = (224,224)
        return applications.VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)
    elif(model_type == 'inception'):
        preferred_size = (299,299)
        if (preferred_size_matches_scale()):
            return applications.InceptionV3(include_top=False, weights='imagenet')
        else:
            return applications.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    elif(model_type == 'xception'):
        print('Using Xception')
        return applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    elif(model_type == 'resnet50'):
        preferred_size = (224,224)
        return applications.ResNet50(include_top=False, weights='imagenet')
  
def compile_model(model, output_optimizer=False):
    learning_rate = float(args.learning_rate)
    opt_arg = args.optimizer.lower()
    optimizer = None
    if (opt_arg == 'sgd'):
        optimizer=optimizers.SGD(lr=learning_rate)
    elif (opt_arg == 'adam'):
        optimizer=optimizers.Adam(lr=learning_rate)
    elif (opt_arg == 'rmsprop'):
        optimizer=optimizers.RMSprop(lr=learning_rate)
    else:
        print(args.optimizer+' IS NOT A VALID OPTIMIZER')

    if(output_optimizer):
        print('Using '+opt_arg+" optimizer")    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

def train_and_save_full_model(base_model, features):
    top_model = create_top_model(features["train_bottleneck_features"], features["train_class_data"]["classes"])
    full_model = train_full_model(base_model, top_model, features)
    save_model_with_metadata(full_model, features["validation_class_data"])

def create_top_model(train_data, classes):
    top_model = Sequential()
    input_shape=train_data.shape[1:]

    if(model_type == 'vgg16'):
        top_model.add(Flatten(input_shape=input_shape))
        top_model.add(Dense(4096, activation='relu'))
        #top_model.add(Dropout(0.5))  
        top_model.add(Dense(4096, activation='relu'))
        #top_model.add(Dropout(0.5))  
    elif (model_type == 'inception' or model_type == 'xception'):
        print("using the inception/xception top model")
        top_model.add(GlobalAveragePooling2D(input_shape=input_shape))
        #top_model.add(Dropout(0.5)) 
        # Extra from other site
        #top_model.add(Dense(1024, activation='relu'))
        #top_model.add(Dropout(0.5)) 
    elif (model_type == 'resnet50'):
         top_model.add(Flatten(input_shape=input_shape))
         top_model.add(Dense(4096, activation='relu'))         
    else:
        print('NO MATCHING TOP MODEL FOUND!!!!!!!!!')


    # top_model.add(Flatten(input_shape=train_data.shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))    

    top_model.add(Dense(len(classes), activation='softmax', name='predictions'))
    print("class_data len: "+ str(len(classes)))

    compile_model(top_model, True)
    return top_model

def create_json_metadata(class_data):
    return json.dumps({
        'classes': class_data["classes"],
        'image_size': {
            'width': img_width,
            'height': img_height
            }
    }).encode('utf8')

def get_classweights(class_map):
    weights = dict()
    for i in range(len(class_map)):
        row = class_map[i]
        for j in range(len(row)):
            if(row[j] == 1):
                if (j not in weights):
                    weights[j] = 1
                else:
                    weights[j] += 1
    return weights

def get_class_weight_import(class_map):
    class_number = get_classweights(class_map)

    # Code from here down borrowed from this project:
    # https://github.com/Arsey/keras-transfer-learning-for-oxford102
    total = np.sum(class_number.values())
    max_samples = np.max(class_number.values())
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.

    return class_weight

def generate_image_predictions(class_map, dir_type, data_dir, datagen, model):
    total_batches = len(class_map) // batch_size
    print(dir_type+" Images Size : " + str(total_batches))

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    predictions =model.predict_generator(
        generator, total_batches, verbose=1)
    #np.save(open('bottleneck_features_train.npy', 'w'),
    #        bottleneck_features_train)
    return predictions

def train_full_model(base_model, top_model, features):
    train_data = features["train_bottleneck_features"] #np.load(open('bottleneck_features_train.npy'))
    validation_data = features["validation_bottleneck_features"] #np.load(open('bottleneck_features_validation.npy'))
    train_class_data = features["train_class_data"]
    validation_class_data = features["validation_class_data"]
  
    class_weight = None
    if(args.use_class_weight==True):
        class_weight = get_class_weight_import(train_class_data["class_map"])
  

    def custom_save_model(model, filepath):
        full_model = create_full_model(base_model, model)
        save_model_with_metadata_to_path(full_model, train_class_data, filepath)

    save_best_only=args.save_best_model
    save_checkpoints = args.save_checkpoints
    checkpoint_dir = check_for_checkpoint_dir()
    model_save_callback = RLCPModelCheckpoint(checkpoint_dir+'/save.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=save_best_only, save_weights_only=False, mode='auto', period=1, custom_save_model=custom_save_model)
    
    callbacks = None
    if(save_best_only or save_checkpoints):
        callbacks = [model_save_callback]
    
    top_model.fit(train_data, train_class_data["class_map"],
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_class_data["class_map"]),
              class_weight = class_weight,
              callbacks=callbacks
              )

    full_model = create_full_model(base_model, top_model)
    
    #test_predictions(full_model, validation_class_data)

    # if (save_best_only):
    #     top_model = model_save_callback.get_best_model()


    return full_model

def create_full_model(base_model, top_model):
    full_model = base_model
    for layer in top_model.layers:
        full_model = Model(inputs= full_model.input, outputs= layer(full_model.output))
    compile_model(full_model)
    return full_model
    
def test_predictions(full_model, validation_class_data):
    preds = generate_image_predictions(validation_class_data["class_map"], "validation", validation_data_dir, get_datagen(), full_model)
    correct = 0
    for i in range(len(preds)):
        prediction = preds[i]
        classes = []
        bestClass = None
        expected_class = '' 
        for j in range(len(prediction)):
            if validation_class_data["class_map"][i][j] == 1:
                expected_class = validation_class_data["classes"][j]
            predClass = {
                "class": validation_class_data["classes"][j],
                "probability": float(prediction[j])
            }
            if (bestClass == None or bestClass["probability"]< predClass["probability"]):
                bestClass = predClass
                bestClass["expected"] = validation_class_data["class_map"][i][j]
                classes.append(predClass)
        correct += bestClass["expected"]
        print("Ground truth: "+str(expected_class)+", Output: "+str(bestClass["class"]))

        # if preds[i] == validation_class_data["class_map"][i]:
        #     correct += 1
    print ('correct '+str(correct)+ ' out of '+str(len(preds))+' - '+str((float(correct)/float(len(preds)))*100)+'%'  )

    

def check_for_destination():
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

def check_for_checkpoint_dir():
    check_for_destination()
    checkpoint_dir = os.path.join(destination_dir,'checkpoints')
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    return checkpoint_dir

def save_model_with_metadata(model,classes):
    save_model_with_metadata_to_path(model, classes, full_model_path)

def save_model_with_metadata_to_path(model,classes, filepath):
    check_for_destination()
    # Save the model to start
    save_model(model, filepath)

    # Load the H5 file   
    f = h5py.File(filepath, mode='r+') # Open file in read/write (r+) mode

    # print("classes")
    # print(classes)
    # print("height")
    # print(img_height)
    # print("width")
    # print(img_width)

    # Set the attributes
    f.attrs['metadata'] = create_json_metadata(classes)

    #Save the H5 file
    f.flush()
    f.close()


main()
