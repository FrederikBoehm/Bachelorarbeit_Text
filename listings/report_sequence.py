class ReportSequence(Sequence):

    def __init__(self, X, y):

        dataset = zip(X, y)
        self.X, self.y = self.__groupByTimesteps(dataset)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (np.array(self.X[idx]), np.array(self.y[idx]))

    def __groupByTimesteps(self, dataset):

        print('Grouping feature vectors by time steps...')

        grouped_reports = {}

        for report_X, report_y in dataset:
            time_steps = len(report_X)

            if time_steps > 0 and len(report_X[0]) > 0:
                if not time_steps in grouped_reports:
                    grouped_reports[time_steps] = {
                        'X': [],
                        'y': []
                    }

                grouped_reports[time_steps]['X'].append(report_X)
                grouped_reports[time_steps]['y'].append(report_y)

        batches_X = []
        batches_y = []

        for key in grouped_reports.keys():
            batches_X.append(grouped_reports[key]['X'])
            batches_y.append(grouped_reports[key]['y'])

        return (batches_X, batches_y)