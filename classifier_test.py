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

binder_model_params = pickle.load(open('./binder_params.pkl','rb'))

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss), jax.nn.softmax(logits)


class BinderClassifier(hk.Module):

    def __init__(self, drop_rate, *args, **kwargs):
        super().__init__(name="BinderClassifier")
        self.drop_rate = drop_rate


    def __call__(self, features, training=False):
        logits1 = jax.nn.gelu(hk.Linear(8)(features))
        binder_logits = hk.Linear(2)(logits1)
        return binder_logits

def binder_classification_fn(input_features, training):
    model = BinderClassifier(0.1)(
        input_features,
        training=training
    )
    return model

rng = jax.random.PRNGKey(43)
binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)


def get_loss_fn(binder_model_params, key, features, labels):
    labels = jnp.array(labels, dtype=jnp.float32)
    logits = binder_classifier.apply(binder_model_params, key, features, training=True)
    binder_loss, prob = softmax_cross_entropy(logits, labels)
    loss = binder_loss.mean()
    return loss, prob



def train_step(binder_model_params, key, features, labels):
    (loss, prob), grads = jax.value_and_grad(get_loss_fn, has_aux=True)(binder_model_params, key, features, labels)
    grads = norm_grads_per_example(grads, l2_norm_clip=0.1)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, prob


def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


scheduler = optax.linear_schedule(0.0, 1e-4, 100, 0)

# Combining gradient transforms using `optax.chain`.
gradient_transform = optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    optax.scale(-1.0) #lr-coeff
)


n_devices = jax.local_device_count()
replicated_binder_model_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), binder_model_params)


opt_state = gradient_transform.init(replicated_binder_model_params)
global_step = 0

loss_list = []
for e in range(10):
      random_perm = np.random.permutation(np.arange(X_train.shape[0]))
      for j in random_perm:
            jax_key, subkey = random.split(jax_key)
            features = jnp.array(X_train)[j,:][None,] #[1,30]
            labels = (np.eye(2)[Y_train.astype(np.int32)])[j]
            labels = jnp.array(labels[None,])
            loss, grads, prob = jax.pmap(train_step, in_axes=(0,None,0,0), axis_name='model_ax')(replicated_binder_model_params, subkey, features, labels)
            global_step += 1
            updates, opt_state = gradient_transform.update(grads, opt_state)
            replicated_binder_model_params = optax.apply_updates(replicated_binder_model_params, updates)
            loss_list.append(loss)
            if (global_step) % 100 == 0:
                print(e+1, global_step, np.mean(loss_list)) 
                print(labels, prob)
                loss_list = []
