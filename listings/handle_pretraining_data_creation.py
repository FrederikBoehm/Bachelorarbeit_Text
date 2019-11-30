def _handlePretrainingDataCreation(files, start_index, end_index, output_dir):
    process_id = os.getpid()
    print(f'Process {process_id} handling files from {start_index} to {end_index}')

    FLAGS = flags.FLAGS
    FLAGS.output_file = output_dir + "/" + "tf_examples.tfrecord"
    for index, single_file in enumerate(files, start_index):

        print(f'Processing {single_file}')
        
        tokenizer = FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        input_files = []
        input_files.extend(tf.io.gfile.glob(single_file))

        logging.info("*** Reading from input files ***")
        for input_file in input_files:
            logging.info("  %s", input_file)

        rng = random.Random(FLAGS.random_seed)
        instances = create_training_instances(
            input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
            rng)

        output_files = (FLAGS.output_file + str(index)).split(",")
        logging.info("*** Writing to output files ***")
        for output_file in output_files:
            logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, output_files)

    print(f'Process {process_id} finished processing.')