pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

train_scores, valid_scores = validation_curve(pipeline,
                                              X_train,
                                              y_train,
                                              param_name="knn__n_neighbors",
                                              param_range=param_range,
                                              cv=5,
                                              n_jobs=-10,
                                              verbose=5)