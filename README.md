# Architectural elements classifier

This project contains the code for the IBM Advanced Data Science Capstone course. You will find thre AI/ML models that predict which is the main architectural element in an image (e.g. a column). 

![Picture of the website generated for the project where users can upload their images and obtain the prediction](./frontend.png])

[Slides](./arch_elements_pitch.pdf)

## Dataset

* Architectural Heritage Elements image dataset
 * Author: Jose Llamas
 * License: Creative Commons Attribution
* 10253 imag
 * 10 categories (altar, aspe, bell tower, column, dome…)
 * 128x128 color

## Models
We have implemented 3 models

* SVM classifier with HOG transformation
* CNN model
* Mobilenet V2 with custom classification output

## Usage

* python > 3.8.5
* python package requirements in the `requirements.txt` file.
* Recommended to have a GPU with cuda and tensorflow configured
* You can run the [`model_training_routines.ipynb`](./model_training_routines.ipynb) file to train and save the models
* If you want to train them in a more automated way, you can use the classes in `model_train` and call them from your application

Run the application (*use flask only for development purposes*):

```bash
FLASK_RUN_PORT=8080 FLASK_APP=arch_elements/model_deployment/app.py flask run --host=0.0.0.0
```

This application implements thre things:

* REST endpoint to receive an image and return the prediction
* Website wher you can upload any image and it returns the predicted element
* Feedback process to help improve the model

![Potential architecture of the application](./arch_elements_architecture.png])

## License 

`Architectural elements identifier` by 8vicat is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 