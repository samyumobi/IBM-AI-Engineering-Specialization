# Keras and Deep Learning Libraries
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Which ofthe following statements is correct?
> 
> 1 / 1 point 
> 

      Keras is a high-level API that facilitates fast development and quick prototyping of deep learning models. 
> 
>  Keras and PyTorch are both supported by Google and are being actively used at Google for both research and production needs. 
> 
>  Among TensorFlow, PyTorch, and Keras, Keras is the most popular library and is mostly used in production of deep learning models. 
> 
>  PyTorch normally runs on top of a low-level library such as TensorFlow. 
> 
>  TensorFlow is the cousin of the Torch framework, which is in Lua, and supports machine learning algorithms running on GPUs in particular. 
> 
> Check
> 
> Correct
> 
> Correct.
> 
>  2.Question 2
> 
> Both TensorFlow and PyTorch are high level APIs for building deep learning models. They provide limited control over the different nodes and layers in a network. If you are seeking more control over a network, then Keras is the right library.
> 
> 1 / 1 point 
> 
>  True 
> 

      False 
> 
> Check
> 
> Correct
> 
> Correct.
> 
>  3.Question 3
> 
> There are three model classes in the Keras library, the Sequential model, the Dense model, and the Model class used with the functional API.
> 
> 1 / 1 point 
> 
>  True 
> 

      False 
> 
> Check
> 
> Correct
> 
> Correct.
> 
>  4.Question 4
> 
> Which of the following codes creates the followig neural network using the Keras library?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8fEfHr0IEem6HRJVmJRoIA_f103b76d1ee8e63b1b41040b3e475c2e_neural_network_final_exam_hidden_layers.png?expiry=1594944000000&hmac=SCIw-zTKAM_JQWbd1tb-I7ROJDGnqQnnWR95QoOcFi0)
> 
> 1 / 1 point 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> model = Sequential()
> 
> model.Dense(add(8, activation='relu', input_shape=(4,)))
> 
> model.Dense(add(5, activation='relu'))
> 
> model.Dense(add(5, activation='relu'))
> 
> model.Dense(add(1))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
> model = Sequential()
> 
> model.Dense(add(8, activation='relu', input_shape=(8,)))
> 
> model.Dense(add(5, activation='relu'))
> 
> model.Dense(add(5, activation='relu'))
> 
> model.Dense(add(1))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> model = Sequential()
> 
> model.add(Dense(8, activation='relu', input_shape=(8,)))
> 
> model.add(Dense(5, activation='relu'))
> 
> model.add(Dense(5, activation='relu'))
> 
> model.add(Dense(1))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 

     model = Sequential()
     
     model.add(Dense(8, activation='relu', input_shape=(4,)))
 
     model.add(Dense(5, activation='relu'))
 
     model.add(Dense(5, activation='relu'))
 
     model.add(Dense(1))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
>  <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
>  
> model = Sequential()
> 
> model.add_Dense(5, activation='relu', input_shape=(4,)))
> 
> model.add_Dense(8, activation='relu'))
> 
> model.add_Dense(4, activation='relu'))
> 
> model.add_Dense(1))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre> 
> 
> Check
> 
> Correct
> 
> Correct.
> 
>  5.Question 5
> 
> If a model can be saved using the Keras library, which of following methods is the correct method to do so?
> 
> 1 / 1 point 
> 
>  model.model_save() 
> 

      model.save() 
> 
>  model.save_model() 
> 
>  model.pickle() 
> 
>  You cannot save a model with the Keras library 
> 
> Check
> 
> Correct
> 
> Correct
>
> -- https://www.coursera.org/learn/introduction-to-deep-learning-with-keras/exam/W7GPq/keras-and-deep-learning-libraries/attempt?redirectToCover=true#Tunnel Vision Close
