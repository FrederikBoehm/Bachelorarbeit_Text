train_sequence = ReportSequence(X_train, y_train)
validation_sequence = ReportSequence(X_valid, y_valid)

model = Sequential()

forward_layer = LSTM(300, return_sequences=True)
backward_layer = LSTM(300, go_backwards=True, return_sequences=True)

model.add(Bidirectional(forward_layer,
                        backward_layer=backward_layer,
                        input_shape=(None, feature_vector_dimension)))
model.add(Bidirectional(LSTM(30, return_sequences=True),
                        backward_layer=LSTM(30, go_backwards=True, return_sequences=True)))
model.add(Bidirectional(LSTM(3, return_sequences=True),
                        backward_layer=LSTM(3, go_backwards=True, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

accuracy_measurement = AccuracyMeasurement(model,
                                           train_sequence,
                                           validation_sequence,
                                           accuracy_results_cv)

model.fit_generator(train_sequence,
                    epochs=20,
                    callbacks=[accuracy_measurement],
                    validation_data=validation_sequence)