for index in range(0, num_train_steps + 1):

    if [x for x in pretrained_checkpoints_list if re.search(f'.*model\.ckpt-{index}\..*', x)]:
        is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.compat.v1.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=output_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=None,
                per_host_input_for_training=is_per_host))

        checkpoint = f'{checkpoints_path}/model.ckpt-{index}'
        result = _evaluateModel(checkpoint,
                                bert_config,
                                run_config,
                                input_files,
                                max_seq_length,
                                num_train_steps,
                                num_warmup_steps,
                                learning_rate,
                                train_batch_size,
                                eval_batch_size,
                                max_eval_steps, 
                                max_predictions_per_seq)
        result["checkpoint"] = checkpoint
        result["step"] = index
        output_df = output_df.append(result, ignore_index=True)
        logging.info("***Eval results***")
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))