# ================================================================================
# ML Sessions #1 - MNIST sin librerias
# ================================================================================
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# FUNCIONES UTILES
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y_one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(labels.size), labels] = 1.0
    return y_one_hot


# ----- funciones de activacion red neuronal ----- #
def relu(z):
    return (z > 0).astype(np.float32)

def relu_derivate(z):
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Estabilidad numérica
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ---- funcion de perdida ----- #
def cross_entropy_loss(y_true, y_pred ):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)            # Evitar log(0) que da -inf y rompe el entrenamiento, tambien se evita log(1) que da 0 por estabilidad numérica
    return - np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ---- evaluacion ----- #
def accuracy_score(y_true_labels, y_pred_probabilities):
    if y_true_labels.ndim > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)

    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    return np.mean(y_true_labels == y_pred_labels)

# ----- redes neuronales ----- #
def forward_pass(X, W1, b1, W2, b2):
    # primera cara
    Z1 = X @ W1 + b1    
    A1 = relu(Z1)

    # segunda capa
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    cache = (X, Z1, A1, Z2, A2)

    return A2, cache



def backward_pass(Y, cache, W2):

    X, Z1, A1, Z2, A2 = cache
    m = X.shape[0]

    dZ2 = A2 - Y

    # gradientes para W2 y b2
    dW2 = (A1.T @ dZ2) / m
    dB2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # propagar gradiente a la capa oculta
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivate(Z1)

    # gradientes para W1 y B1
    dW1 = (X.T @ dZ1) / m
    dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, dB1, dW2, dB2
    

print("help functions created")



# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# CARGAR LOS DATOS
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('data created')

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# PROCESAMIENTO
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# Aplanar imagenes (28, 28) -> (784,)
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')

# Normalizar los datos (0-255) -> (0-1)
X_train /= 255.0
X_test /= 255.0

# One-hot encode las etiquetas
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)

print('preprocessing done')


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# CONFIGUCIÓN
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

input_size = 784    # 28x28=784 pixeles
hidden_size = 128   # neuronas en la capa oculta
output_size = 10    # dígitos del 0 al 9

lr = 0.01           # tasa de aprendizaje
epochs = 10         # número de épocas para entrenar
batch_size = 64     # tamaño del batch para entrenamiento


# ----- inicializar pesos y bias ----- #
W1 = np.random.randn(input_size, hidden_size) * 0.01
B1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
B2 = np.zeros((1, output_size))

print('weights and bias done')


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# TRAIN LOOP
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯


num_train = X_train.shape[0] # numero de muestras -> 60k
loss_history = []

print('starting training \n\n\n')

for epoch in range(1, epochs + 1):

    # shuffle (mezclar)
    indices = np.random.permutation(num_train)
    X_train_shuffle = X_train[indices]
    Y_train_shuffle = y_train[indices]

    epoch_loss = 0.0

    num_batches = num_train // batch_size

    for i in range(0, num_batches):

        batch_start = i*batch_size
        batch_end = batch_start + batch_size

        X_batch = X_train_shuffle[batch_start:batch_end]
        Y_batch = Y_train_shuffle[batch_start:batch_end]

        # forward pass
        Y_pred, cache = forward_pass(X_batch, W1, B1, W2, B2)

        # calcular loss del batch 
        loss = cross_entropy_loss(Y_batch, Y_pred)
        epoch_loss += loss * X_batch.shape[0] # multiplicamos por m porque el loss ista promediado sobre el bache

        # backward pass
        dW1, dB1, dW2, dB2 = backward_pass(Y_batch, cache, W2)

        # actualizar parametros del gradient decent
        W1 = W1 - lr*dW1
        W2 = W2 - lr*dW2
        B1 = B1 - lr*dB1
        B2 = B2 - lr*dB2

    # loss promedio durante toda la epocha 
    epoch_loss = epoch_loss / num_train
    loss_history.append(epoch_loss)

    # evaluar en train y test sets
    y_pred_train, _ = forward_pass(X_train, W1, B1, W2, B2)
    y_pred_test, _ = forward_pass(X_test, W1, B1, W2, B2)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# Plot loss
plt.figure()
plt.plot(loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.grid(True)
plt.show()


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# EVALUACION
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

y_pred, _ = forward_pass(X_test, W1, B1, W2, B2)
acc = accuracy_score(y_pred, y_test)

print("\nFinal Test Accuracy:", round(acc, 4))






