import re
import os
import random
import tarfile
import urllib
from torchtext.legacy import data


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar.gz'
    dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))

class Assign0(TarDataset):

    dirname = 'assignment1'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):

            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'neg_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'pos_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
            with open(os.path.join(path, 'neu_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'neutral'], fields) for line in f]
            print(len(examples))
            with open(os.path.join(path, 'neg_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'pos_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
            with open(os.path.join(path, 'neu_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'neutral'], fields) for line in f]
        super(Assign0, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        path = "assignment1/Dataset0"
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        dev_index = 2798

        print(f'Training Len  {len(examples[:dev_index])} ')
        print(f'Validation Len  {len(examples[dev_index:])} ')

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


class Assign1(TarDataset):

    dirname = 'assignment1'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):

            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'neg_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'pos_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
            print(len(examples))
            with open(os.path.join(path, 'neg_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'pos_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(Assign1, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        path = "assignment1/Dataset1"
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        dev_index = 12664

        print(f'Training Len  {len(examples[:dev_index])} ')
        print(f'Validation Len  {len(examples[dev_index:])} ')

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))

class Assign2(TarDataset):

    dirname = 'assignment1'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):

            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, '1_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '1'], fields) for line in f]
            with open(os.path.join(path, '2_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '2'], fields) for line in f]
            with open(os.path.join(path, '3_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '3'], fields) for line in f]
            with open(os.path.join(path, '4_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '4'], fields) for line in f]
            with open(os.path.join(path, '5_train.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '5'], fields) for line in f]
            print(len(examples))
            with open(os.path.join(path, '1_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '1'], fields) for line in f]
            with open(os.path.join(path, '2_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '2'], fields) for line in f]
            with open(os.path.join(path, '3_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '3'], fields) for line in f]
            with open(os.path.join(path, '4_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '4'], fields) for line in f]
            with open(os.path.join(path, '5_valid.txt'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, '5'], fields) for line in f]

        super(Assign2, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        path = "assignment1/Dataset2"
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        dev_index = 2797

        print(f'Training Len  {len(examples[:dev_index])} ')
        print(f'Validation Len  {len(examples[dev_index:])} ')

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))
