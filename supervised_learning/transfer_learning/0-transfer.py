#!/usr/bin/env pyhton3
"""Module: 0-transfer"""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Preprocesses CIFAR-10 data for Keras applications.
    """
    # Standardize input channel values using application-specific preprocessing
    X_p = K.applications.ResNet50.preprocess_input(X)

    # Convert scalar class labels to one-hot vectors
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p

if __name__ == "__main__":
    # Load data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)
    print("Data loaded and scaled.")

    input_layer = K.Input(shape=(32, 32, 3))
    # Scale up small 32x32 images so receptive fields function properly
    resized_input = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224)))(input_layer)

    # Load base model and freeze it
    base_model = K.applications.ResNet50(
        weights='imagenet',
        input_tensor=resized_input,
        include_top=False,
        pooling='average'  # automatically appends a GlobalAveragePooling2D
    )
    base_model.trainable = False

    # Feature extractor (Input - Base model)
    feature_extractor = K.Model(inputs=input_layer, outputs=base_model.output)

    # Extract features
    train_features = feature_extractor.predict(X_train_p, batch_size=64, verbose=1)
    test_features = feature_extractor.predict(X_test_p, batch_size=64, verbose=1)

    # Build custom Top Classifier on pre-computed features
    classifier_input = K.Input(shape=feature_extractor.output_shape[1:])
    x = K.layers.BatchNormalization()(classifier_input)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    output_layer = K.layers.Dense(10, activation='softmax')(x)
    classifier = K.Model(inputs=classifier_input, outputs=output_layer)

    # Compile and train
    classifier.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(
            train_features, Y_train_p, validation_data=(test_features, Y_test_p),
            epochs=15, batch_size=64, verbose=1
    )

    # FINE TUNING?
    # 5. Assemble and save full, callable end-to-end model
    final_input = K.Input(shape=(32, 32, 3))
    features = feature_extractor(final_input)
    final_output = classifier(features)

    full_model = K.Model(inputs=final_input, outputs=final_output)

    # Needed for evaluation
    full_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    full_model.save('cifar10.h5')