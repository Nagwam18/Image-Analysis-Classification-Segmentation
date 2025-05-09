import matplotlib.pyplot as plt

def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax[0].plot(epochs, acc, label='Training Accuracy', color='blue')
    ax[0].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss plot
    ax[1].plot(epochs, loss, label='Training Loss', color='blue')
    ax[1].plot(epochs, val_loss, label='Validation Loss', color='orange')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.suptitle('U-Net Performance')
    plt.show()
