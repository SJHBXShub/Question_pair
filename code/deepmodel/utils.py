#s this file is used to define kinds of utility functions
import keras

class EarlyStopWhenValLossLessThanExpect(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=5, expect_value = 0.3):
        self.expect_value = expect_value
        self.monitor = monitor
        self.patience = patience
        self.stopped_epoch = 0
        self.flag_expect = False

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        if current <= self.expect_value:
            self.patience -= 1

        if self.patience == 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.flag_expect = True

    def on_train_end(self, logs=None):
        print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class MyFunction(object):
    # Read Embedding File
    def read_embedding(filename):
        embed = {}
        for line in open(filename,encoding='utf-8'):
            line = line.strip().split()
            embed[str(line[0])] = list(map(float, line[1:]))
        print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
        return embed

    @staticmethod
    def save_vector(file_path, vector, mode='w'):
        """
        Save vector on disk
        :param file_path: vector file path
        :param vector: a vector in List type
        :param mode: mode of writing file
        :return: none
        """
        file = open(file_path, mode, encoding='utf-8')
        for value in vector:
            file.write(str(value) + "\n")
        file.close()
        return

def aveResult(result_array_list):
    final = sum(result_array_list)/len(result_array_list)
    final = pd.DataFrame(final)
    final.to_csv('result_.txt', index = None, header = None)
