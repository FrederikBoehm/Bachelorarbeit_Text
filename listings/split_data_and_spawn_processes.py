def _splitDataAndSpawnProcesses(input_files, output_dir, bert_object):
    cpu_cores = cpu_count()
    processes = []
    print(f'Detected {cpu_cores} cores, splitting dataset...')
    for index in range(cpu_cores):
        start_index = int(len(input_files) / cpu_cores) * index
        end_index = int(len(input_files) / cpu_cores) * (index + 1)
        sObject = slice(start_index, end_index)
        splitted_files_list = input_files[sObject]
        print(splitted_files_list)

        process = Process(target=_handlePretrainingDataCreation, args=(splitted_files_list,
                                                                       start_index,
                                                                       end_index,
                                                                       output_dir,
                                                                       bert_object))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()