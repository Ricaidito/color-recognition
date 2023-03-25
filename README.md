# color-recognition

Machine Learning model written in Python using PyTorch and scikit-learn capable of classifying the colors in any supplied image. Powered by a dataset (`colors.csv`) with color information

## How to run it

It is recommended to use a virtual environment to avoid problems when installing the packages. To create it just run `py -m venv venv`.

1. Install all the required packages and dependencies with `pip install -r requirements.txt`
2. Run the file `model.py` to compile, train and save the ML model.
3. Finally, to test the model just run the `main.py` file.

**Only images with `.jpg` extension are supported.**

_Note: The larger the image, the longer it will take to process the colors._
