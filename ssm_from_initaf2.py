import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import string
import optax
import csv
import argparse
import json
from dateutil import parser
import jax
import warnings
warnings.filterwarnings("ignore")
import jax.numpy as jnp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from alphafold.common import protein
from alphafold.common.protein import from_pdb_string
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data import parsers
from alphafold.data.parsers import parse_a3m
from alphafold.relax.amber_minimize import *
from alphafold.model.tf.data_transforms import *
from alphafold.model.all_atom import *
from alphafold.model import quat_affine
from alphafold.model import utils
from alphafold.common import confidence
import haiku as hk
from alphafold.data import mmcif_parsing
from typing import Tuple, List
import pandas as pd
from train_utils import *
from typing import Tuple
import functools
import time
import pickle
import torch
from jax import random
from jax import jit

load_optimizer = False
home_path = '/home/aditya20/experimentsWaf2'
scratch_path = '/net/scratch/aditya20/af2exp'

jax_key = jax.random.PRNGKey(0)

class BinderClassifier(hk.Module):

    def __init__(self, drop_rate, *args, **kwargs):
        super().__init__(name="BinderClassifier")
        self.drop_rate = drop_rate

    def _rbf(self, D):
        D_min, D_max, D_count = 2., 22., 32
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        RBF = jnp.exp(-((D[:,None] - D_mu[None,:]) / D_sigma)**2)
        return RBF

    def _jnp_softmax(self, x, axis=-1):
        unnormalized = jnp.exp(x - jax.lax.stop_gradient(x.max(axis, keepdims=True)))
        return unnormalized / unnormalized.sum(axis, keepdims=True)


    def __call__(self, input_features, training=False):
         
        Ca = input_features['structure_module_final_atom_positions'][:,1,:] #[L, 3]
        Ca_mask = input_features['structure_module_final_atom_mask'][:,1] #[L]
        mask = input_features['peptide_mask']
        Ca_mask_2D = Ca_mask[:,None]*Ca_mask[None,:]
        mask_2D = (mask[:,None])*(1-mask[None,:])
        distances = jnp.sqrt(jnp.sum((Ca[:,None,:] - Ca[None,:,:])**2, -1)+1e-8) #[L1,L2]
        inter_distances = distances*mask_2D*Ca_mask_2D + (1-mask_2D*Ca_mask_2D)*10000.0
        closest_distances = jnp.min(inter_distances,-1) #[L1]
        features = jnp.sum(closest_distances*mask*Ca_mask, -1, keepdims=True)/jnp.sum(mask*Ca_mask, -1, keepdims=True)
        features_binned = self._rbf(features)
        

        mask_2D_symm = mask_2D + mask_2D.T
        pae_probs = self._jnp_softmax(input_features['predicted_aligned_error']['logits']) 
        pae_probs = jnp.sum(Ca_mask_2D[:,:,None]*mask_2D_symm[:,:,None]*pae_probs, (-2, -3))/jnp.sum(Ca_mask_2D*mask_2D_symm)
        
        #features = jnp.concatenate([features_binned[0,], pae_probs]) 
        features = pae_probs
        
        logits1 = jax.nn.gelu(hk.Linear(8)(features))
        binder_logits = hk.Linear(2)(logits1)

        return binder_logits, features
    
binder_model_params = pickle.load(open('/net/scratch/aditya20/af2exp/model/binder_params_pae.pkl','rb'))

def binder_classification_fn(input_features, training):
    model = BinderClassifier(0.1)(
        input_features,
        training=training
    )
    return model

rng = jax.random.PRNGKey(43)
binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.asarray(loss), jax.nn.softmax(logits)

def get_loss_fn(binder_model_params, key, processed_feature_dict, structure_flag,labels):
    #labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    logits, features = binder_classifier.apply(binder_model_params, key, processed_feature_dict, training=True)
    binder_loss, prob = softmax_cross_entropy(logits, labels)
    binder_loss_mean = binder_loss.mean()
    loss = binder_loss_mean
    return loss, (predicted_dict, prob, features)


def train_step(binder_model_params, af2_model_params, key, batch, structure_flag):
    #binder_model_params, af2_model_params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", model_params)
    (loss, (predicted_dict, prob, features)), grads = jax.value_and_grad(get_loss_fn, has_aux=True)(binder_model_params, key, batch, structure_flag)
    grads = norm_grads_per_example(grads, l2_norm_clip=0.1)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict, prob, features

def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads

scheduler = optax.linear_schedule(0.0, 1e-2, 200, 0)

# Combining gradient transforms using `optax.chain`.
gradient_transform = optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0*0.01) #lr-coeff
)

n_devices = jax.local_device_count()
replicated_binder_model_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), binder_model_params)

if load_optimizer:
    opt_state = pickle.load(open(optimizer_state_path, "rb")) 
    global_step = int(np.load(global_step_path))
else:
    opt_state = gradient_transform.init(replicated_binder_model_params)
    global_step = 0

loss_log = {}
grad_log = {}

training_data = glob.glob('/net/scratch/aditya20/af2exp/ssm_initaf2_training/*.npz')
structure_flag = True

score_labels = {}
with open(f'{home_path}/valid_binder_data.sc') as f:
    for line in f:
        words = line.strip().split()
        pdb = words[0]
        beneficial = words[1]
        neutral = words[3]
        
        if beneficial == 'True' or neutral == 'True':
            score_labels[pdb] = 1
        else:
            score_labels[pdb] = 0
            
for e in range(10):
    
    loss_log[e] = {}
    t0=time.time()
    
    print(f'Epoch {e+1} ---------')
    
    for j, npzfile in enumerate(training_data):
        
        tmp = np.load(npzfile,allow_pickle=True)
        input_features = tmp['arr_0'].item()
        
        jax_key, subkey = random.split(jax_key)
        
        pdb = npzfile.split('__pred')[0].split('tmp_')[1]
        labels = np.array([score_labels[pdb]],np.int32)
        label = np.eye(2)[labels]
        #input_features['labels'] = label
        
        batch = {}
        
        for k,v in input_features.items():
            if type(v) != dict:
                batch[k] = v
            else:
                for k1,v1 in v.items():
                    #print(k1)
                    batch[k+'_'+k1] = v1

        del batch['ptm']
        del batch['max_predicted_aligned_error']
        del batch['distogram_bin_edges']
        del batch['masked_msa_logits']
        
        batch['plddt'] = batch['plddt'][:,None,None]
        batch['peptide_mask'] = batch['peptide_mask'][:,None,None]
        #batch['distogram_bin_edges'] = batch['distogram_bin_edges'][:,None,None]
        batch['predicted_lddt_logits'] = batch['predicted_lddt_logits'][:,:,None]
        batch['structure_module_final_atom_mask'] = batch['structure_module_final_atom_mask'][:,:,None]
        batch['predicted_aligned_error'] = batch['predicted_aligned_error'][:,:,None]
        batch['experimentally_resolved_logits'] = batch['experimentally_resolved_logits'][:,:,None]
        
        #batch['labels'] = batch['labels'][:,None]
        for k,v in batch.items():
            print(k,v.shape)
            
        loss, grads, predicted_dict, prob, features = jax.pmap(train_step, in_axes=(0,None,0,None,None), axis_name='model_ax')(replicated_binder_model_params, subkey, batch, structure_flag,label)
        
        global_step += 1
        updates, opt_state = gradient_transform.update(grads, opt_state)
        replicated_binder_model_params = optax.apply_updates(replicated_binder_model_params, updates)
        
        
        print(f'Structure step: {global_step}, binder_loss: {loss}, label: {labels}, binder_prob: {prob}')
        
        