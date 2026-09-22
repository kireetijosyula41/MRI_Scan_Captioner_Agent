# Generative AI Final Project: Kireeti Josyula***

This project includes a Streamlit MRI classifier, a Grad-CAM visualization, and an OpenAI-generated report. The app runs the saved classifier with PyTorch.

**Requirements**

The app uses `streamlit-app.py`, `torch_classifier.py`, and the existing `brain_tumor.h5` weights. The HDF5 file is read directly with `h5py`; running the app does not require TensorFlow or Keras.

Use Python 3.11 or newer for the app. Install its dependencies with:

```sh
python3 -m pip install -r requirements.txt
```

The training dataset is available at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. It is not needed to run the app with an uploaded image.

**Model Generation**

`model.ipynb` is the original TensorFlow/Keras training notebook. It is retained as a record of how `brain_tumor.h5` was trained; the app loads those weights into the matching PyTorch architecture at startup. Rerunning the notebook requires its original training dependencies.

The PyTorch weight mapping and Grad-CAM can be checked with `python3 -m unittest discover -s tests -v`.

**Evaluation**

Run `python3 evaluation.py` to evaluate the saved model on `braintumordata/Testing`. The script uses the app's preprocessing and PyTorch classifier, then writes `metrics.json`, `confusion_matrix.png`, and `predictions.csv` in the project directory. Use `--testing-dir`, `--model`, `--output-dir`, or `--batch-size` to change the defaults.

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

This opens the app, where you can upload an MRI image and view the class probabilities, Grad-CAM overlay, and generated report.
