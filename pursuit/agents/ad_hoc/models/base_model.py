import _pickle as pickle

from keras.models import load_model


class BaseModel(object):
    def __init__(self, model_size):
        self.model_size = model_size
        self.model = None

    def save(self, filename):
        if self.model is not None:
            self.model.save(filename + '.model')
        d = dict(self.__dict__)
        d.pop('model')
        f = open(filename, 'wb')
        pickle.dump(d, f)
        f.close()

    @classmethod
    def load(cls, filename):
        model = load_model(filename + '.model')
        f = open(filename, 'rb')
        attrs = pickle.load(f)
        f.close()
        obj = cls(attrs['model_size'])
        for key, value in attrs.items():
            setattr(obj, key, value)
        obj.model = model
        return obj