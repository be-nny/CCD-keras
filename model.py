from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.optimizers import schedules
from keras import layers, losses, optimizers
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def load_data(file_name, sheet):
    data = pd.read_excel(file_name, sheet_name=sheet)
    print(" -- Sheet Loaded -- ")

    print(" \n-- Sheet Info -- ")
    print(data.info())

    print(" \n-- Sheet Description -- ")
    print(data.describe())

    print(" \n-- Checking Data -- ")
    print(data.isna().sum())

    for i in data.isna().sum():
        if i != 0:
            print("WARNING: Data contains NaN values")
            return None
    print("DATA OK")
    return data

def pre_process(data):
    # pre-processing
    # dropping the 'default credit card' column from the test and train set
    # removing the index 0 row from the spreadsheet file
    data = data.drop(0)

    Y = pd.get_dummies(data["Y"]).iloc[:, 1]
    X = data.drop(labels=["Y"], axis=1)

    # plotting bias before
    print(f"Y: {Y.value_counts()}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    Y.value_counts().plot(kind="bar", color=["red", "blue"])
    plt.title("Bias before Random Under Sampling")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    for i, count in enumerate(Y.value_counts()):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.xticks(rotation=0)

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    X_res, Y_res = rus.fit_resample(X, Y)

    # plotting bias after
    print(f"Y: {Y_res.value_counts()}")
    plt.subplot(1, 2, 2)
    Y_res.value_counts().plot(kind="bar", color=["red","blue"])
    plt.title("Bias after Random Under Sampling")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    for i, count in enumerate(Y_res.value_counts()):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.show()

    # creating the training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def sequential(X_train, X_test, Y_train, Y_test):
    NB_EPOCH = 60
    BATCH_SIZE = 128
    VERBOSE = 1
    VALIDATION_SPLIT = 0.4

    model = Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Learning Rate Scheduler
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(lr_schedule),
        metrics=['accuracy', f1_score_metric]
    )

    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=NB_EPOCH,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping]
    )

    score = model.evaluate(
        X_test, Y_test,
        verbose=0
    )

    # making predictions
    Y_pred_proba = model.predict(X_test).ravel()
    Y_pred = (Y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)

    print(f"Loss: {score[0]}, Score: {score[1]}, Accuracy: {accuracy}, Precision: {precision}, F1 Score: {f1}, Recall: {recall}")

    # plotting f1 score
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(history.history['f1_score_metric'])
    plt.plot(history.history['val_f1_score_metric'])
    plt.title('Model F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plotting loss
    plt.subplot(1,3,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plotting loss
    plt.subplot(1,3,3)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plotting confusion matrix
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
                )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def f1_score_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

if __name__ == "__main__":
    credit_card_defaults = load_data("CCD.xls", "Data")
    if credit_card_defaults is not None:
        X_train_set, X_test_set, Y_train_set, Y_test_set = pre_process(credit_card_defaults)
        sequential(X_train_set, X_test_set, Y_train_set, Y_test_set)
