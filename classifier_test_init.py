from sklearn import datasets
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import numpy as np
import haiku as hk
import jax
import pickle
import optax
import random
from jax import random

X, Y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=123)

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                   jnp.array(X_test, dtype=jnp.float32),\
                                   jnp.array(Y_train, dtype=jnp.float32),\
                                   jnp.array(Y_test, dtype=jnp.float32)

samples, features = X_train.shape
classes = np.unique(Y)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

jax_key = jax.random.PRNGKey(0)

binder_model_params = pickle.load(open('/net/scratch/aditya20/af2exp/model/binder_params_pae_and_dist.pkl','rb'))

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss), jax.nn.softmax(logits)


class BinderClassifier(hk.Module):

    def __init__(self, drop_rate, *args, **kwargs):
        super().__init__(name="BinderClassifier")
        self.drop_rate = drop_rate


    def __call__(self, features, training=False):
        logits1 = jax.nn.gelu(hk.Linear(8, w_init=hk.initializers.VarianceScaling(scale=1.0))(features))
        binder_logits = hk.Linear(2, w_init=hk.initializers.VarianceScaling(scale=1.0))(logits1)
        return binder_logits

def binder_classification_fn(features, training):
    model = BinderClassifier(0.1)(
        features,
        training=training
    )
    return model

rng = jax.random.PRNGKey(43)
binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)



binder_params = binder_classifier.init(
    rng,
    features=jnp.array(np.random.randn(30)),
    training=True
)
print('done')
pickle.dump(binder_params, open('./binder_params.pkl', "wb"))

