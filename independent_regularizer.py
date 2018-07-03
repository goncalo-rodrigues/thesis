from keras.regularizers import Regularizer
from keras.losses import kullback_leibler_divergence
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scipy.stats import entropy


class IndepedentRegularizer(Regularizer):
    def __init__(self, sizes, weight=0.1):
        self.sizes = sizes
        self.idx = get_indices(sizes)
        self.weight = weight

    def __call__(self, x):
        loss = 0.
        marginalized_vars = []
        for i, size in enumerate(self.sizes):
            marginalized = K.sum(tf.gather(x, self.idx[i], axis=-1), axis=-1)
            # marginalized = marginalized / K.sum(marginalized) # this should not be needed
            marginalized_vars.append(marginalized)
        products = K.stack([a for a in combine(marginalized_vars, self.sizes, lambda x1, x2: x1*x2)])
        loss += self.weight*kullback_leibler_divergence(x, products)
        return loss

def combine(values, sizes, combiner):
    if len(values) == 1:
        result = []
        for i in range(sizes[0]):
            result.append(values[0][..., i])
        return result
    result = []
    for i in range(sizes[0]):
        token = values[0][..., i]
        for combination in combine(values[1:], sizes[1:], combiner):
            result.append(combiner(token, combination))
    return result

def get_indices(sizes):
    import numpy as np
    total = 1
    for size in sizes:
        total *= size
    curr_total = total
    prev = [0]
    result = []
    for size in sizes:
        curr_total = curr_total // size
        new_prev = []
        indices = [[] for _ in range(size)]
        for start in prev:
            for var in range(size):
                new_prev.append(start+var*curr_total)
                for idx in range(start+var*curr_total, start+(var+1)*curr_total):
                    indices[var].append(idx)
        result.append(np.array(indices))
        prev = new_prev
    return result

def get_idx(values, sizes):
    result = 0
    for value, size in zip(values, sizes):
        result = result*size + value
    return result

import numpy as np
np.random.seed(101)
n = 50
probs = np.array(combine(np.array([[1,2], [2,1], [4, 3], [2, 3],[4,1], [2,2]]), (2,2,2,2,2,2), lambda x1,x2: x1*x2))
probs = probs / sum(probs)
dataset = np.zeros((n, len(probs)))
dataset[np.arange(n), np.random.choice(range(len(probs)), p=probs, size=n, replace=True)] = 1

# model = Sequential()
# model.add(Dense(units=len(probs), activation='softmax', input_dim=1, activity_regularizer=IndepedentRegularizer((2,2,2), weight=1.)))
# model.compile(optimizer='sgd', loss='categorical_crossentropy')
# model.fit(np.zeros((n, 1)), dataset, batch_size=10, verbose=False)
# print(entropy(probs, model.predict(np.array([[0]]))[0]))
# model = Sequential()
# model.add(Dense(units=len(probs), activation='softmax', input_dim=1))
# model.compile(optimizer='sgd', loss='categorical_crossentropy')
# model.fit(np.zeros((n, 1)), dataset, batch_size=10, verbose=False)
# print(entropy(probs, model.predict(np.array([[0]]))[0]))

reg = IndepedentRegularizer((2,2,2,2,2,2), 10.)
freqs = K.constant((1+np.sum(dataset, axis=0)) / (n+len(probs)))
prob_tensor = K.variable(np.sum(dataset, axis=0) / n)
prob_tensor = prob_tensor / tf.reduce_sum(prob_tensor)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(reg(prob_tensor)+kullback_leibler_divergence(freqs, prob_tensor))
init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    for step in range(1000):
      session.run(train)
      print("step", step,
            "entropy:", entropy(probs, session.run(prob_tensor)))

    print(session.run(prob_tensor))
    print(session.run(freqs))
print(entropy(probs, (1+np.sum(dataset, axis=0)) / (n+len(probs))))
# TODO: compute (count) p(x), p(y),... individually and compare with real distribution