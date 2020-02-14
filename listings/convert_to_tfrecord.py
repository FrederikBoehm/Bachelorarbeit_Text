def _convertToTfrecord(examples, vocab_file, output_dir, output_file, max_seq_length):
    processor = ReportProcessor()
    label_list = processor.get_labels()
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_file)

    print('Converting files to tfrecord.')
    file_based_convert_examples_to_features(
            examples, label_list, max_seq_length, tokenizer, output_file_path)
    print(f'Finished tfrecord creation at {output_file_path}')