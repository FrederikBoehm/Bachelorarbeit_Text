bert_config = BertConfig.from_json_file(bert_config_file)

is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.compat.v1.estimator.tpu.RunConfig(
    cluster=None,
    master=None,
    model_dir=output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    keep_checkpoint_max=30,
    tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=None,
        per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=init_checkpoint,
    learning_rate=learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,
    use_one_hot_embeddings=False)

# If TPU is not available, this will fall back to normal Estimator on CPU
# or GPU.
estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size)

train_input_fn = input_fn_builder(
    input_files=input_files,
    max_seq_length=max_seq_length,
    max_predictions_per_seq=max_predictions_per_seq,
    is_training=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)