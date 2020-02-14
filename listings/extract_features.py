model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False)

# If TPU is not available, this will fall back to normal Estimator on CPU
# or GPU.
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    predict_batch_size=batch_size)

input_fn = input_fn_builder(
    features=features, seq_length=max_seq_length)

vectorized_text_segments = []
for result in estimator.predict(input_fn, yield_single_examples=True):
    layer_output = result["layer_output_0"]
    feature_vec = [
        round(float(x), 6) for x in layer_output[0].flat
    ]
    vectorized_text_segments.append(feature_vec)