import argparse
import sys
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageOps
from tabulate import tabulate
import psutil
import os
import requests
from matplotlib import pyplot as plt
import math
import zipfile


# CONSTANTS
# Terminal colors to format the training output beautifully
# I hosted any necessasry data on dropbox in case runs without it
GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

TEST_ZIP_URL = "https://www.dropbox.com/scl/fi/o9fuc4jmw3d1yicgkx7ox/fashion-mnist_test.zip?rlkey=6rzb65g5ch1r8556aosor2s24&st=ky1pqen3&dl=1"
TEST_ZIP_FILE = "fashion_mnist_test.zip"

TRAIN_ZIP_URL = "https://www.dropbox.com/scl/fi/1fvdvjypzgoibft9qcck3/fashion-mnist_train.zip?rlkey=ld56rq24irrxnnntdknp4wyhj&st=25uvypz8&dl=1"
TRAIN_ZIP_FILE = "fashion_mnist_train.zip"

TRAIN_FILE = "fashion-mnist_train.csv"
TEST_FILE = "fashion-mnist_test.csv"
MODEL_FILE = "model_weights.npz"
IMAGE_FILE = "example.jpg"

REQUIRED_TRAIN_FILES = [TRAIN_FILE]
REQUIRED_TEST_FILES = [TEST_FILE, MODEL_FILE, IMAGE_FILE]

FASHION_MNIST_LABELS = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover",
    3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
    7: "Sneaker", 8: "Bag",  9: "Ankle boot"}


#--Entry Point--------------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train a 3-layer neural network or run predictions and tests. This project works best with the fashion MNIST dataset. For testing purposes you can use a mini-version of the MNIST dataset, which is limited to only 20,000 samples. If you have time to spare train the regular dataset, much better results. Note: training data, pretrained model, exmaple image, and test data will be downloaded on the first run. (very small)"
    )


    parser.add_argument(
        "-t", "--train",
        nargs='?',
        type=str,
        const="regular",
        default=None,
        help="Train the network on either the mini dataset or the regular dataset (default: regular). mini: faster training, high risk of overfitting (~85%% accuracy) regular: medium speed training, very accurate model (~89-90%% accuracy)"
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=500,
        help="Number of iterations (Default: 500)"
    )

    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=1.0,
        help="Learning rate (default: 1)"
    )

    parser.add_argument(
        "-b", "--beta",
        type=float,
        default=0.9,
        help="Beta (default: 0.9)"
    )

    parser.add_argument(
        "-p", "--predict",
        nargs='*',
        metavar=("IMAGE", "WEIGHTS"),
        default=None,
        help="Uses a trained weights to do a prediction on a provided image. Pass nothing for defaults (example image and pretrained weights will be downloaded (~1mb)), or pass IMAGE (png, jpg, jpeg, webp) or/and model weights (Optional)"
    )


    parser.add_argument(
        "-tst", "--test",
        nargs="?",
        type=str,
        const='model_weights.npz',
        default=None,
        help="Opens a window with 20 example images from the fashion-mnist test dataset and runs a trained model on them. You can provide your own model.npz or leave empty to check the pretrained model."
    )



    args = parser.parse_args()

    if len(sys.argv) < 2:
        print("No arguments provided. Use -h for help.")
        sys.exit(1)


    # Download only the required files for either training or testing
    elif args.train is not None:

        train(args.train, args.iterations, args.alpha, args.beta)


    elif args.test is not None:
            ensure_files_exist(REQUIRED_TEST_FILES, TEST_ZIP_URL, TEST_ZIP_FILE )
            if args.test is not None:
                if args.test.strip().endswith('.npz'):
                    test(args.test)
                else:
                    sys.exit("Error: please provide a valid model weights (.npz) file")



    elif args.predict is not None:
        ensure_files_exist(REQUIRED_TEST_FILES, TEST_ZIP_URL, TEST_ZIP_FILE)
        predict(args.predict)
    else:
        print("Wrong arguments provided. Use -h for help.")
        sys.exit(1)



def train(args_train: str, iterations: int, ALPHA: float, BETA: float) -> None:
    """Trains a neural network on a provided dataset."""
    if args_train in ["mini","regular"]:
        ensure_files_exist(REQUIRED_TRAIN_FILES, TRAIN_ZIP_URL, TRAIN_ZIP_FILE )
        data_type = args_train


        Y_val, X_val, Y_train, X_train, mean, std = load_dataset("fashion-mnist_train.csv", data_type)
        W1, b1, W2, b2, W3, b3 = train_gradient_descent(X_train, X_val, Y_train, Y_val, iterations, ALPHA, BETA)
        filepath = input("Output path without extension ('n' to not save model): ").strip()
        if filepath != "n":
            save_model(f"{filepath}.npz", W1, b1, W2, b2, W3, b3, mean, std)
        elif filepath == "n":
            sys.exit("Model not saved. Exiting")

    else:
        sys.exit("Error: -t expects mini (mini dataset) or regular (regular dataset)")



def predict(args_predict: list) -> None:
        """Run a trained model on a specified image. Default runs the pretrained model on a real life sample image."""

        if args_predict and not (1 <= len(args_predict) <= 2):
            sys.exit("--predict requires either 1 or 2 values")
        # if only one argument is provided next to --predict
        elif len(args_predict) == 1:
            if args_predict[0].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_path = args_predict[0]
                model_path = "model_weights.npz"
            elif args_predict[0].lower().endswith((".npz")):
                image_path = "example.jpg"
                model_path = args_predict[0]
            else:
                sys.exit(f"Error: '{args_predict[0]}' is not a valid file format. Provide an image (.png, .jpg, .jpeg, .webp) or model weights (.npz).")

        # if only --predict (uses default)
        elif len(args_predict) == 0:
                image_path = "example.jpg"
                model_path = "model_weights.npz"
        else:
            if args_predict[1].lower().endswith('.npz') and args_predict[0].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_path = args_predict[0]
                model_path = args_predict[1]

            elif args_predict[0].lower().endswith('.npz') and args_predict[1].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_path = args_predict[1]
                model_path = args_predict[0]

            else:
                sys.exit("Invalid input: please provide a valid image or a valid image and model weights.")

        if not os.path.exists(image_path):
            sys.exit(f"Error: {image_path} not found")

        if not os.path.exists(model_path):
            sys.exit(f"Error: {model_path} not found")

        W1, b1, W2, b2, W3, b3, mean, std = load_model(model_path)
        processed_image = preprocess_img(mean, std, image_path)
        label, confidence = feedforward(W1, b1, W2, b2, W3, b3, processed_image)
        display_prediction(label, confidence, processed_image, image_path)



def ensure_files_exist(req_files, url, output) -> None:
    """Ensures that all datasets, pretrained model, and test images are downloaded and available in the current working directory."""

    # If all the files exist, dont download
    if all(os.path.exists(f) for f in req_files):
            print("Files found in current working directory")
            return
    else:

        print(f"Downloading files from source...")
        extract_name = output+".part"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            total_chunks_for_now = 0
            if total_size == 0:
                total_size = total_chunks_for_now + 20
            with open(extract_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    total_chunks_for_now += len(chunk)
                    clear_terminal()
                    # Homemade progress bar from temu
                    print(f"\rDownloading data: {'█'*int(total_chunks_for_now/total_size*100)}{'░'*(100-(int(total_chunks_for_now/total_size*100)))}", f"{total_chunks_for_now/total_size*100:.2f}%")
                    f.write(chunk)

            os.replace(extract_name, output)
            print("Download complete. Extracting...")


            with zipfile.ZipFile(output) as zf:
                zf.extractall(".")

            print(f"\n Successfully downloaded all files\n")

        except (Exception, KeyboardInterrupt) as e:
            # If there is an error while downloading delete the half downloaded zip file
            if os.path.exists(extract_name):
                os.remove(extract_name)

            print(f"\nError downloading files: {e}")
            sys.exit(1)



def colorize(val: any, is_good: bool) -> str:
    """Returns the metrics value colored in green if its improving, or red if not."""

    if is_good:
        color = GREEN
    else:
        color = RED

    return f"{color}{val}{RESET}"



def load_dataset(train_path: str, data_type: str) -> tuple:
    """Load and process the dataset."""

    print("Loading dataset.")
    try:
        data = pd.read_csv(train_path)
        data = np.array(data)
        np.random.shuffle(data)
    except Exception as E:
        sys.exit(f"Error: {E}")

    # I Transpose the data so the lables become the first row (which makes them much easier to extract)
    # the first 1000 examples will be used for validation check
    # I find the average of the pixel colors and subtract that from every pixel, then divide everything by the std of X_train.
    # This standarizes the pixels so average pixels have a value of 0, anything dakrer than average is negative, and anything brighter than average is positive
    # Take only 20,000 examples if its the mini set
    if data_type == "mini":
        data = data[:20000]

    m, n = data.shape
    data_dev = data[0:1000].T
    label_dev = data_dev[0].astype(np.int8)
    X_dev_pre = data_dev[1:n].astype(np.float32) / 255.0


    data_train = data[1000:m].T
    label_train = data_train[0].astype(np.int8)
    X_train_pre = data_train[1:n].astype(np.float32) / 255.0

    mean = np.mean(X_train_pre)
    std = np.std(X_train_pre) + 1e-8 #1e-8 is a very small number to prevent division by 0

    # np.ascontiguousarray helps place the data in the same block of memory after it has been transposed (which i did in the previous step) helps cpu cache performance

    X_dev = np.ascontiguousarray((X_dev_pre - mean) / std, dtype="float32")
    X_train = np.ascontiguousarray((X_train_pre - mean) / std, dtype="float32")
    print("Dataset loaded.")

    return label_dev, X_dev, label_train, X_train, mean, std



def load_test_dataset(mean: float, std: float) -> tuple:
    """Loads the test data and downloads it if it doesnt exist."""

    # data the model hasnt seen before
    test_data = 'fashion-mnist_test.csv'

    print("Loading test dataset.")
    try:
        data = pd.read_csv(test_data)
        data = np.array(data)
        np.random.shuffle(data)
    except Exception as E:
        sys.exit(f"Error: {E}")

    # transpose to fit the trained models inputs
    data = data.T
    m,n = data.shape
    Y_test = data[0].astype('int8')
    X_test = data[1:m].astype(np.float32) / 255.0
    X_test = np.ascontiguousarray((X_test - mean) / std, dtype="float32")

    print("Loaded test dataset")
    return Y_test, X_test



def init_parameters() -> tuple:
    """Initializes the paramters of the weights and biases."""

    # Float 32 for blazingly fast operations
    W1 = (np.random.randn(128,784) * np.sqrt(2.0/784)).astype(np.float32)
    b1 = np.zeros((128,1), dtype=np.float32)
    W2 = (np.random.randn(64,128) * np.sqrt(2.0/128)).astype(np.float32)
    b2 = np.zeros((64,1), dtype=np.float32)
    W3 = (np.random.randn(10,64) * np.sqrt(2.0/64)).astype(np.float32)
    b3 = np.zeros((10,1), np.float32)
    print("Initializing parameters")
    return W1, b1, W2, b2, W3, b3



def ReLU(Z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) is an activation function used in hidden layers to add non-linearity.
"""

    return np.maximum(0, Z)



def d_ReLU(Z: np.ndarray) -> np.ndarray:
    """The derivate of the ReLU function."""

    return (Z > 0).astype(np.float32)



def Softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax is an activatioon function used in the output layer. It Turns the linear outputs of the neural network into a probability vector with values that sum up to 1.0."""

    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=0, keepdims=True)



def forward_propagation(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray, X: np.ndarray) -> tuple:
    """Feeds the weights and biases into the neural network."""

    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = Softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3



def one_hot(Y: np.ndarray) -> np.ndarray:
    """One hot encoding, creates an empty (zero) matrix and places 1 only at the location of the correct label."""

    m = Y.size
    one_hot_y = np.zeros((10, Y.size), dtype='float32')

    # np.arange(Y.size) creates an array going from 0 - Y.size. np allows to iterate over lists inside a list
    # so first it would be one_hot_y[0,Y] (which is the first row, to the location of the value of Y) and set it to 1.

    one_hot_y[Y, np.arange(Y.size)] = 1
    return one_hot_y



def backward_propagation(Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray, Z3: np.ndarray, A3: np.ndarray,W2: np.ndarray, W3: np.ndarray, X: np.ndarray, Y_one_hot: np.ndarray,) -> tuple:
    """Backwards propagation function is what makes neural networks learn.
        It back tracks to which neuron made mistakes and adjusts its weights accordingly."""

    m = Y_one_hot.shape[1]


    # Here there should be a softmax derivative, but because the Cross-Entropy loss and Softmax derivative cancel out algebraically, we get this simple subtraction
    dZ3 = A3 - Y_one_hot
    dW3 = 1.0/m * dZ3.dot(A2.T)
    db3 = 1.0/m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * d_ReLU(Z2)
    dW2 = 1.0/m * dZ2.dot(A1.T)
    db2 = 1.0/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = 1.0/m * dZ1.dot(X.T)
    db1 = 1.0/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW3, db3, dW2, db2, dW1, db1



def init_velocities(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> tuple:
    """Creates records of the velocites of the gradients"""

    v_dW1 = np.zeros_like(W1)
    v_db1 = np.zeros_like(b1)
    v_dW2 = np.zeros_like(W2)
    v_db2 = np.zeros_like(b2)
    v_dW3 = np.zeros_like(W3)
    v_db3 = np.zeros_like(b3)

    return v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3



def update_parameters(W1, b1, W2, b2, W3, b3, dW3, db3, dW2, db2, dW1, db1, v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3, ALPHA: float, BETA: float) -> tuple:
    """Updates the parameters using gradient descent and momentum based on the errors found during backward propagation

        ALPHA is the learning rate, essentially dictating the weight of the error correction or how big of a step to take.
        Lower ALPHA means longer training time, but better learned data. Vice Versa

        BETA is known as the Momentum Decay Coefficient. It tells the network how much it should rely on old gradients, and how much to rely on new ones.
        its set to 0.9 by default, which means the network retains the values of 90% of old gradients, and only adds in 10% from new gradients
        """
    v_dW1 = BETA * v_dW1 + (1.0 - BETA) * dW1
    v_db1 = BETA * v_db1 + (1.0 - BETA) * db1
    v_dW2 = BETA * v_dW2 + (1.0 - BETA) * dW2
    v_db2 = BETA * v_db2 + (1.0 - BETA) * db2
    v_dW3 = BETA * v_dW3 + (1.0 - BETA) * dW3
    v_db3 = BETA * v_db3 + (1.0 - BETA) * db3

    W3 -= ALPHA * v_dW3
    b3 -= ALPHA * v_db3
    W2 -= ALPHA * v_dW2
    b2 -= ALPHA * v_db2
    W1 -= ALPHA * v_dW1
    b1 -= ALPHA * v_db1

    return W1, b1, W2, b2, W3, b3, v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3



def get_prediction(A3: np.ndarray) -> np.ndarray:
    """Finds the value with the highest percetange in the final prediction, which is the label key."""

    return np.argmax(A3, axis=0)



def get_accuracy(A3: np.ndarray, Y: np.ndarray) -> float:
    """Finds the accuracy of the prediction using the real labels"""

    prediction = get_prediction(A3)
    return (np.sum(prediction == Y) / Y.size)*100



def get_loss(A3: np.ndarray, Y_one_hot: np.ndarray) -> float:
    """Quantifies how wrong the neural networks predictions are compared to their actual values."""

    m = Y_one_hot.shape[1]

    # Clipping because log(0) = infinity. So I set the mininmum value in A3 as a very tiny number (1^-15)
    A3_clipped = np.clip(A3, 1e-15, 1.0 - 1e-15)

    loss = -1/m * np.sum(Y_one_hot * np.log(A3_clipped))
    return loss



def calc_gradient_norm(dW1: np.ndarray, dW2: np.ndarray, dW3: np.ndarray) -> tuple:
    """Shows the magnitude of the gradient matrices through the Frobenius norm."""

    dW1_norm = np.linalg.norm(dW1)
    dW2_norm = np.linalg.norm(dW2)
    dW3_norm = np.linalg.norm(dW3)
    return dW1_norm, dW2_norm, dW3_norm



def save_model(save_path, W1, b1, W2, b2, W3, b3, mean: float, std: float) -> None:
    """Saves the weights and biases inside a compressed .npz file, which packs numpy arrays into a zipped archive and reserves floating point percision."""

    print(f"Saving model to {save_path}")
    np.savez_compressed(save_path,
                        W1=W1, b1=b1,
                        W2=W2, b2=b2,
                        W3=W3,b3=b3,
                        mean=mean, std=std)
    print(f"model successfully saved to {save_path}!")



def load_model(filepath="model_weights.npz") -> tuple:
    """Loads the weights and biases from an already existing model."""

    print("Loading model")

    try:
        parameters = np.load(filepath)
    except Exception as e:
        sys.exit(f"Error: {e}")

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    mean = parameters['mean']
    std = parameters['std']
    print("Model loaded.")
    return W1, b1, W2, b2, W3, b3, mean, std



def clear_terminal() -> None:
    """Helps clear the terminal using ANSII escape sequence right before printing information again. Enchancing look and understanding"""

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()



def print_metrics(i, train_acc, train_loss, val_acc, val_loss, dW1_norm, dW2_norm, dW3_norm, time_sec, training_time, PREV_METRICS) -> None:
    """Prints the metrix in a fancy tabulate table, alongside colored acc and loss values to display improvement."""

    cpu_usage = psutil.cpu_percent(interval=None)



    # Compare the current metrics to the previous iteration, decided if they're improving or not.
    t_acc_good = train_acc >= PREV_METRICS.get('train_acc', train_acc)
    t_loss_good = train_loss <= PREV_METRICS.get('train_loss', train_loss)
    v_acc_good = val_acc >= PREV_METRICS.get('val_acc', val_acc)
    v_loss_good = val_loss <= PREV_METRICS.get('val_loss', val_loss)
    formatted_time = time.strftime("%M:%S", time.gmtime(time.time() - training_time))


    PREV_METRICS = {
        'train_acc': train_acc, 'train_loss': train_loss,
        'val_acc': val_acc, 'val_loss': val_loss
    }


    metrics = [
        ["Iteration", i, "dW1 Norm", f"{dW1_norm:.4f}"],
        ["Train Acc", colorize(f"{train_acc:.2f}%", t_acc_good), "dW2 Norm", f"{dW2_norm:.4f}"],
        ["Train Loss", colorize(f"{train_loss:.4f}", t_loss_good), "dW3 Norm", f"{dW3_norm:.4f}"],
        ["Val Acc", colorize(f"{val_acc:.2f}%", v_acc_good), "Time (per iter)", f"{time_sec:.4f}s"],
        ["Val Loss", colorize(f"{val_loss:.4f}", v_loss_good), "AVG Cpu Usage", f"{ORANGE}{cpu_usage:.1f}%{RESET}"]
    ]


    print(f"{GREEN}Green means the metric is improving{RESET}", "|")
    print(f"{RED}Red means the metric is not improving{RESET}", "|")
    print(tabulate(metrics, tablefmt="fancy_grid"))
    print(f"Elapsed training time: {formatted_time} minutes")

    return PREV_METRICS



def get_label(A3):
    "returns the label class and the confidence rate for a provided prediction"

    prediction = get_prediction(A3)[0]
    confidence = float(np.max(A3, axis=0)[0]) * 100
    label = FASHION_MNIST_LABELS[prediction]


    return label, confidence



def preprocess_img(mean, std, img_path="example.jpg") -> np.ndarray:
    """Preprocess the image and make it the same format as the training images."""

    # L is grayscale
    img = Image.open(img_path).convert('L')


    img = img.resize((28, 28), Image.Resampling.LANCZOS)


    img = ImageOps.invert(img)

    img_array = (((np.array(img, dtype=np.float32) / 255.0) - mean) / std).astype(np.float32)

    processed_image = img_array.reshape(784, 1) 

    return processed_image



def feedforward(W1, b1, W2, b2, W3, b3, processed_image) -> tuple:
    """makes a prediction using the trained weights and biases on the processed image.
        Simply run a forward propagation once, the prediction is A3."""

    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, processed_image)
    label, confidence = get_label(A3)
    return label, confidence



def test(model_path="model_weights.npz") -> None:
    """Run either a provided model or the pretrained one on images from the testing dataset and create a window using matplotlib to display predictions."""


    test_accuracy = 0

    W1, b1, W2, b2, W3, b3, mean, std = load_model(model_path)

    Y_test, X_test = load_test_dataset(mean, std)
    _,n = X_test.shape

    # unpack each image out of the test data and append them to a list
    # extracts all rows of the i-th column
    image_list = [X_test[:, i:i+1] for i in range(X_test.shape[1])]


    real_labels = [FASHION_MNIST_LABELS[i] for i in Y_test]

    fig, axes = plt.subplots(4, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n):
        predicted_label, confidence = feedforward(W1, b1, W2, b2, W3, b3, image_list[i])
        real_label = real_labels[i]
        resized_image = image_list[i].reshape(28,28)

        if real_label == predicted_label:
            test_accuracy += 1.0
            color = "green"
        else:
            color = "red"
         # Show the 20 first image as examples
        if i < 20:
            axes[i].imshow(resized_image, cmap="gray")
            axes[i].set_title(f"Pred: {predicted_label} | Real: {real_label}\nConfidence: {confidence:.2f}%", color=color, fontsize=10)
            axes[i].axis('off')

    print(f"Testing complete! Your model has a test accuracy of {(test_accuracy/n)*100:.1f}%")
    fig.suptitle(f"Test accuracy: {(test_accuracy/n)*100:.1f}%\n" f"(Showing first 20 images out of {n} total test samples)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()



def display_prediction(label: str, confidence: float, processed_image, img_path) -> None:
    """Using matplotlib to render a small window to display the image, prediction, and confidence"""

    # Turn the processed image from a (784,1) vector matrix back to a 28x28 matrix
    processed_image = processed_image.reshape(28, 28)

    user_image = Image.open(img_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(user_image)
    ax1.set_title("What you see:")
    ax1.axis('off')

    ax2.imshow(processed_image, cmap="gray")
    ax2.set_title("what the model sees: ")
    ax2.axis("off")

    fig.suptitle(f"Label: {label} | Confidence: {confidence:.2f}%", fontsize=16, fontweight="bold")


    plt.tight_layout()
    plt.show()



def save_metrics(METRICS_HISTORY: dict[str, list[float]], output_file="training_metrics.png") -> None:
    """Save the graph of the training metrics."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the loss graph
    ax1.plot(METRICS_HISTORY["iterations"], METRICS_HISTORY["train_loss"], label="Train Loss", color="blue")
    ax1.plot(METRICS_HISTORY["iterations"], METRICS_HISTORY["val_loss"], label="Val Loss", color="orange", linestyle="--")
    ax1.set_title("Loss over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot the accuracy graph
    ax2.plot(METRICS_HISTORY["iterations"], METRICS_HISTORY["train_acc"], label="Train Acc", color="green")
    ax2.plot(METRICS_HISTORY["iterations"], METRICS_HISTORY["val_acc"], label="Val Acc", color="red", linestyle="--")
    ax2.set_title("Accuracy over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, format="png", dpi=300)
    plt.close()
    print(f"Metrics plot successfully saved as {output_file}")



def cosine_annealing(current_iteration: int, total_iterations: int, max_alpha: float, min_alpha: float) -> float:
    """Automatically calculates the best rate of decay for the learning rate using a smooth cosine curve."""

    if current_iteration >= total_iterations:
        return min_alpha

    return min_alpha + 0.5 * (max_alpha - min_alpha) * (1 + math.cos((current_iteration/total_iterations)*math.pi))



def train_gradient_descent(X, X_val, Y, Y_val, iterations: int, ALPHA: float, BETA: float) -> tuple:
    """Trains the neural network using gradient descent."""

    # A dictionary to track the pervious training metrics to know if they're getting better or worse
    # Another dictionary to capture training metrics every 10 iterations to graph them
    PREV_METRICS= {}
    METRICS_HISTORY = {
        "iterations": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    training_time = time.time()

    # Initalize cpu tracking to understand cpu usage
    psutil.cpu_percent(interval=None)
    # Initalize parameters and velocities
    W1, b1, W2, b2, W3, b3 = init_parameters()
    v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3 = init_velocities(W1, b1, W2, b2, W3, b3)
    print("Preparing the engine")
    Y_one_hot = one_hot(Y)
    Y_one_hot_val = one_hot(Y_val)
    for i in range(iterations+1):
        decayed_ALPHA = cosine_annealing(i, iterations, ALPHA, min_alpha=ALPHA/1000.0)
        current_start_time = time.time()
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW3, db3, dW2, db2, dW1, db1 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y_one_hot)
        dW1_norm, dW2_norm, dW3_norm = calc_gradient_norm(dW1, dW2, dW3)
        W1, b1, W2, b2, W3, b3, v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3 = update_parameters(W1, b1, W2, b2, W3, b3 ,dW3, db3, dW2, db2, dW1, db1,v_dW1, v_db1, v_dW2, v_db2, v_dW3, v_db3, decayed_ALPHA, BETA)
        time_per_iteration = time.time() - current_start_time
        # Print training metrics every 10 iterations
        if  i == 0 or i % 10 == 0:
            _,_,_,_,_,A3_val = forward_propagation(W1, b1, W2, b2, W3, b3, X_val)
            train_acc = get_accuracy(A3, Y)
            train_loss = get_loss(A3, Y_one_hot)
            val_acc = get_accuracy(A3_val, Y_val)
            val_loss = get_loss(A3_val, Y_one_hot_val)

            # Record and save the training metrics information every 10 iterations to graph later
            METRICS_HISTORY["iterations"].append(i)
            METRICS_HISTORY["train_loss"].append(train_loss)
            METRICS_HISTORY["val_loss"].append(val_loss)
            METRICS_HISTORY["train_acc"].append(train_acc)
            METRICS_HISTORY["val_acc"].append(val_acc)

            clear_terminal()
            PREV_METRICS = print_metrics(i, train_acc, train_loss, val_acc, val_loss, dW1_norm, dW2_norm, dW3_norm, time_per_iteration, training_time, PREV_METRICS)
            # homemade progress bar from temu again (im very happy with it)
            print(f"Training progress: {'█'*int(i/iterations*100)}{'░'*(100-(int(i/iterations*100)))}", f"({i}/{iterations}) {i/iterations*100:.2f}%")

    print(f"Training complete | {iterations} iterations | {ALPHA} learning rate | {BETA} Beta")
    save_metrics(METRICS_HISTORY)
    return W1, b1, W2, b2, W3, b3



if __name__ == "__main__":
    main()
