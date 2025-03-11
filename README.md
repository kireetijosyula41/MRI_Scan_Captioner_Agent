***Generative AI Final Project: Kireeti Josyula***

Hello, this is the manual guide for my Generative AI Final Project, which was training a MRI brain tumor image classifier and designing a streamlit app
would allow a user to upload an MRI image and use OpenAI's API to generate a detailed caption of the image. 

**Requirements**

The source code for this project are in two files: model.ipynb and streamlit-app.py

IMPORTANT: In order to run the code this project, a version of python that is 3.8 is required. This is due to the tensorflow requirement in the model generation

To download the dataset for this project, navigate to the following website on kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
and download the dataset to your local machine. The file contains I would advise creating a main directory and placing the downloaded dataset file along with 
the model.ipynb and streamlit-app.py files as well as the brain_tumor.h5 file.

Finally, there are many libraries necessary to be installed on your local machine in order to run the code involved in this project

To install these libraries, run the following commands on your terminal or directory where the model.ipynb and streamlit-app.py files are

```
pip install numpy
pip install tensorflow
pip install openai
pip install matplotlib
pip install Pillow=10.4.0
pip install streamlit
pip install pandas
pip install keras
```

These installations should be enough to get the code properly running. 

**Model Generation**

In order to generate the model, the model.ipynb file makes use of the following imports 

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
```

To generate the 'brain_tumor.h5' model, run through each code block in the model.ipynb notebook. Certain file paths in the code will have to be changed for your local machine.

**Streamlit App Running**

Before running the streamlit app, run the command

```
mkdir -p .streamlit
touch .streamlit/secrets.toml
```
on your terminal in your local directory. Within the .toml file, write the following

```
OPENAI_API_KEY = 'your api key here'
```
and fill in the prompt with the OPENAI_API_KEY. This is NECESSARY for the streamlit app to run. If this step is not done, the code will not execute.

To run the streamlit app, navigate to the directory where the streamlit-app.py file is stored and run the command

```streamlit run streamlit-app.py```

This will open a streamlit app tab, where you can upload a MRI brain scan image from the Testing file of the downloaded dataset, and see the returned image caption
