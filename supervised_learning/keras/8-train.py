#!/usr/bin/env python3
""" early stopping, learning rate decay, and model checkpoint"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains the model using mini-batch gradient descent,
    optionally using early stopping, learning rate decay,
    and saving the best model based on validation loss.
    """

    callbacks = []

    # Early stopping callback
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    # Learning rate decay callback
    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch, lr):
            new_lr = alpha / (1 + decay_rate * epoch)
            print(f"\nEpoch {epoch+1}: Learning rate is {new_lr:.6f}")
            return new_lr

        lr_decay = K.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
        callbacks.append(lr_decay)

    # Save best model callback
    if save_best and validation_data is not None and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,      # Only save if val_loss improves
            mode='min',               # Lower val_loss is better
            verbose=0
        )
        callbacks.append(checkpoint)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
