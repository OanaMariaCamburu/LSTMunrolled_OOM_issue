from torchtext import data

import util
import csv


class eSNLI(data.TabularDataset):

    def __init__(self, path, format, fields, **kwargs):
        super().__init__(path, format, fields, skip_header=True, **kwargs)

    urls = []
    dirname = 'esnli'
    name = 'esnli'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.sentence1), len(ex.sentence2))

    @classmethod
    def splits(cls, text_field, label_field, pair_field, parse_field=None, root='../../data',
               train='train.csv', validation='dev.csv', test='test.csv'): 

        all_bad_datapoints = ["2057991743.jpg#2r1c", "7638876050.jpg#0r1c", "7638876050.jpg#4r2n", "7638876050.jpg#4r2e", "7638876050.jpg#4r2c", "112178718.jpg#2r5c", "112178718.jpg#2r5n", "507370108.jpg#4r1e", "507370108.jpg#4r1c", "1153704539.jpg#0r1c", "3398745929.jpg#3r1c", "2753157857.jpg#2r4c", "651277216.jpg#1r1n", "4882073197.jpg#2r3n", "4882073197.jpg#2r3c", "3808935147.jpg#1r1c", "3808935147.jpg#1r1n", "3808935147.jpg#1r1e", "162967671.jpg#0r2c", "2755595842.jpg#2r1e", "4679327842.jpg#1r1c", "4679327842.jpg#1r1n", "5062422406.jpg#3r1c", "5062422406.jpg#3r1e", "5062422406.jpg#3r1n"]
        empty_datapoints = ["7638876050.jpg#4r2n", "7638876050.jpg#4r2e", '7638876050.jpg#4r2c']
        
        
        path = cls.download(root, check='esnli')

        my_args = util.get_args()
        if my_args.sanity:
            train = "dev.csv"
            validation = "dev.csv"
            test = "dev.csv"
        else:
            train = my_args.train_file
            validation = my_args.dev_file
            test = my_args.test_file

        #for sanity checks we will give same train/dev/test and get the first n_data datapoints out, -1 stands for taking all
        n_data = my_args.n_data
        if train == validation and validation == test and n_data > -1:
            print("Using smaller dataset")
            # create a subset of the dataset with the n_data rows
            f = open(path + "/" + train)
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            subset_dataset_file = path + "/" + str(n_data) + "subset_" + train
            util.remove_file(subset_dataset_file)
            g = open(subset_dataset_file, "a")
            writer = csv.writer(g)
            writer.writerow(headers)
            i = 0
            while i < n_data:
                row = next(reader)
                writer.writerow(row.values())
                i += 1
            train = str(n_data) + "subset_" + train
            validation = str(n_data) + "subset_" + validation
            test = str(n_data) + "subset_" + test
            g.close()
            f.close()

        assert(parse_field is None)
        return super(eSNLI, cls).splits(
            path, root, train, validation, test,
            format='csv', fields=[('gold_label', label_field), ('sentence1_binary_parse', None), ('sentence2_binary_parse', None), ('sentence1_parse', None), ('sentence2_parse', None), ('sentence1', text_field), ('sentence2', text_field), ('captionID', None), ('pairID', pair_field), ('label1', None), ('label2', None), ('label3', None), ('label4', None), ('label5', None)],
            filter_pred=lambda ex: (ex.gold_label != '-') and (ex.pairID not in empty_datapoints))
  
    