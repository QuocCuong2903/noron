import cv2 as cv
import numpy as np


def LAYMAU(path):
    img = cv.imread(cv.samples.findFile(path), cv.IMREAD_GRAYSCALE)

    # Edge detection using Canny
    edges = cv.Canny(img, 50, 150)

    # Find contours in the binary image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize geometric features array
    features = np.zeros(18)

    if len(contours) > 0:
        # Assuming the largest contour is the shape of interest
        contour = max(contours, key=cv.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Number of vertices
        vertices = len(approx)

        # Add the number of vertices to the features array
        features[vertices - 3] = 1

    return features


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def SOSANHKHACMANG(x, y):
    return not np.array_equal(x, y)

def TEST(path):
    try:
        ex = LAYMAU(path)
        result_probs = softmax(np.dot(ex, w) + bias)
        print(path, '==> Raw Scores:', np.dot(ex, w) + bias)
        print(path, '==> Probabilities:', result_probs)

        # Get the predicted class
        predicted_class = np.argmax(result_probs)
        print(path, '==> Predicted Class:', predicted_class)

    except Exception as e:
            print("Error during testing:", e)
            predicted_class = -1

    # Print the predicted shape based on the class label
    if predicted_class == 0:
        print(path, '==> HINH TRON')
    elif predicted_class == 1:
        print(path, '==> HINH VUONG')
    elif predicted_class == 2:
        print(path, '==> HINH TAMGIAC')
    else:
        print(path, "==> CHUA BIET HINH NAY LA HINH GI")


# ...

if __name__ == "__main__":
    # Define training data and labels for circle, square, and triangle
    p1 = LAYMAU('./shapes/circle')
    p2 = LAYMAU('./shapes/triangle')
    p3 = LAYMAU('./shapes/square')
    t1 = np.array([[0, 1, 0]], ndmin=2)  # HINH TRON
    t2 = np.array([[0, 0, 1]], ndmin=2)  # HINH VUONG
    t3 = np.array([[1, 0, 0]], ndmin=2)  # HINH TAM GIAC

    # Combine all training data and labels
    p_train = np.vstack((p1, p2, p3))
    t_train = np.vstack((t1, t2, t3))

    # Initialize weights and bias
    np.random.seed(42)  # For reproducibility
    w = np.random.randn(18, 3) * 0.01
    bias = np.zeros((1, 3))
    learning_rate = 0.0001

    w_old = np.full((18, 3), 0)
    bias_old = np.zeros((1, 3))
    lanlap = 0

    max_iterations = 1000
    convergence_threshold = 1e-5

    while lanlap < max_iterations:
        w_old = w.copy()
        bias_old = bias.copy()

        for i in range(len(p_train)):
            a = np.dot(p_train[i], w) + bias
            result_probs = softmax(a)
            e = t_train[i] - result_probs
            w = w + learning_rate * np.outer(p_train[i], e)
            bias = bias + learning_rate * e

        if not SOSANHKHACMANG(w, w_old) and np.array_equal(bias, bias_old):
            print('Converged after {0} iterations'.format(lanlap))
            break

        lanlap += 1

    print('Da hoc xong voi so lan lap la {0}'.format(lanlap))
    print('Ma tran trong so w = ', w)
    print('Bias = ', bias)

    TEST('./shapes/circle/circle2.jpg')
    TEST('./shapes/triangle/white-large-square_2b1c (1).jpg')
    TEST('./shapes/square/square3.png')
