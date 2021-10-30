"""
Predictor interfaces for the Deep Learning challenge.
"""
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from PIL import Image


class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """

    # TODO
    def load_model(self, model_path: str = 'default path'):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """
        #------------------------------------------------------------------------------
        # Get model path
        if model_path == 'default path':         
          import deep_equation
          model_path =  str(deep_equation.__path__[0]) + '/model/model.h5'
          #print(model_path)

        # Load model
        self.model = keras.models.load_model(model_path)
        #self.model.summary()
        return
    
    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """

        #------------------------------------------------------------------------------
        ####### LOAD MODEL #######
        self.load_model()
        
        #------------------------------------------------------------------------------
        ####### PREPROCESSING #######
        size = 224,224

        # Images A
        X1 = np.zeros([len(operators),size[0],size[1],3],dtype = 'float16')
        for i, img in enumerate(images_a):
            img = img.resize(size)
            X1[i,:,:,:] = np.array(img)[:,:,:3].astype('float16')

        # Images B
        X2 = np.zeros([len(operators),size[0],size[1],3],dtype = 'float16')
        for i, img in enumerate(images_b):
            img = img.resize(size)
            X2[i,:,:,:] = np.array(img)[:,:,:3].astype('float16')
        
        # Normalization
        X1 = X1/255.
        X1 = X1 - 0.5
        X1 = X1*2
        X1 = X1.astype('float16')
      
        X2 = X2/255.
        X2 = X2 - 0.5
        X2 = X2*2
        X2 = X2.astype('float16')
        
        # Operator
        dictop = {'+':0,'-':1,'*':2,'/':3}
        Xop = []
        for op in operators:
            Xop.append(dictop[op])
        Xop = np.asarray(Xop)
        Xop = to_categorical(Xop,num_classes=4).astype('uint8')

        #------------------------------------------------------------------------------
        ####### PREDICTION #######
        y_pred = []
        for i in range(len(X1)):
             x1 = X1[i,:,:,:].reshape([1,224,224,3])
             x2 = X2[i,:,:,:].reshape([1,224,224,3])
             xo = Xop[i,:].reshape([1,4])
             y_pred.append(self.model.predict([x1,x2,xo]))
             
        #------------------------------------------------------------------------------
        ####### POST PROCESSING #######
        # Function onehot -> float
        def oh2class(y_pred):
            y_pred_int = [np.argmax(i) for i in y_pred]
            y_pred_int = np.array(y_pred_int)
            classes = [ -9.  ,  -8.  ,  -7.  ,  -6.  ,  -5.  ,  -4.  ,  -3.  ,  -2.  ,
                    -1.  ,   0.  ,   0.11,   0.12,   0.14,   0.17,   0.2 ,   0.22,
                    0.25,   0.29,   0.33,   0.38,   0.4 ,   0.43,   0.44,   0.5 ,
                    0.56,   0.57,   0.6 ,   0.62,   0.67,   0.71,   0.75,   0.78,
                    0.8 ,   0.83,   0.86,   0.88,   0.89,   1.  ,   1.12,   1.14,
                    1.17,   1.2 ,   1.25,   1.29,   1.33,   1.4 ,   1.5 ,   1.6 ,
                    1.67,   1.75,   1.8 ,   2.  ,   2.25,   2.33,   2.5 ,   2.67,
                    3.  ,   3.5 ,   4.  ,   4.5 ,   5.  ,   6.  ,   7.  ,   8.  ,
                    9.  ,  10.  ,  11.  ,  12.  ,  13.  ,  14.  ,  15.  ,  16.  ,
                    17.  ,  18.  ,  20.  ,  21.  ,  24.  ,  25.  ,  27.  ,  28.  ,
                    30.  ,  32.  ,  35.  ,  36.  ,  40.  ,  42.  ,  45.  ,  48.  ,
                    49.  ,  54.  ,  56.  ,  63.  ,  64.  ,  72.  ,  81.  , np.nan  ]
            y_pred_float = np.asarray([classes[aux] for aux in y_pred_int])
            return y_pred_float.astype('float')
        #------------------------------------------------------------------------------
        # Float conversion
        predictions = oh2class(y_pred)
        #print('\n\n#############\n',predictions)
        return list(predictions)

