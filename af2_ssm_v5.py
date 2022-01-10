import os
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
#from absl import flags
import torch
from jax import random
from jax import jit
import tensorflow.compat.v1 as tf1
warnings.filterwarnings("ignore")
flags = tf1.app.flags

home_path = '/home/aditya20/experimentsWaf2'
scratch_path = '/net/scratch/aditya20/af2exp'

load_optimizer = False
optimizer_state_path = '/net/scratch/aditya20/af2exp/model/state_ssm_v5_5799.pkl'
global_step_path     = '/net/scratch/aditya20/af2exp/model/ssm_v5_global_step.npy'
model_weights_path   = '/net/scratch/aditya20/af2exp/model/ssm_v5_5799.pkl'

### BINDER CLASSIFIER LINEAR LAYER WITH DIST AND PAE ###


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
         
        Ca = input_features['structure_module']['final_atom_positions'][:,1,:] #[L, 3]
        Ca_mask = input_features['structure_module']['final_atom_mask'][:,1] #[L]
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
        
        features = jnp.concatenate([features_binned[0,], pae_probs]) 
        logits1 = jax.nn.gelu(hk.Linear(16)(features))
        binder_logits = hk.Linear(2)(logits1)

        return binder_logits, features

batch_size = 1

jax_key = jax.random.PRNGKey(0)
model_name = "model_3_ptm" #no-templates
model_config = config.model_config(model_name)
model_config.data.common.resample_msa_in_recycling = True
model_config.model.resample_msa_in_recycling = True
model_config.data.common.max_extra_msa = 1024
model_config.data.eval.max_msa_clusters = 512
model_config.data.eval.crop_size = 256
model_config.model.heads.structure_module.structural_violation_loss_weight = 1.0
model_config.model.embeddings_and_evoformer.evoformer_num_block = 48

if load_optimizer:
    full_model_params = pickle.load(open(model_weights_path, "rb"))
    binder_model_params, af2_model_params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", full_model_params)
else:
    af2_model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/projects/ml/alphafold/")
    binder_model_params = pickle.load(open('/home/amotmaen/af2exp/af2-fine-tuning/models/binder_params.pkl','rb'))

model_runner = model.RunModel(model_config, af2_model_params)
model_params = hk.data_structures.merge(af2_model_params, binder_model_params)

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

def sigmoid_cross_entropy(logits, labels):
  """Computes sigmoid cross entropy given logits and multiple class labels."""
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  loss = -labels * log_p - (1. - labels) * log_not_p
  return jnp.asarray(loss), jnp.exp(log_p)

def get_loss_fn(model_params, key, processed_feature_dict, structure_flag):
    binder_model_params, af2_model_params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", model_params)
    classifier_dict = {}
    labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    classifier_dict['peptide_mask'] = processed_feature_dict['peptide_mask']
    del processed_feature_dict['peptide_mask']
    del processed_feature_dict['labels']
    predicted_dict, loss = model_runner.apply(af2_model_params, key, processed_feature_dict)
    classifier_dict.update(predicted_dict)
    logits, features = binder_classifier.apply(binder_model_params, key, classifier_dict, training=True)
    binder_loss, prob = softmax_cross_entropy(logits, labels)
    binder_loss_mean = binder_loss.mean()
    loss = jnp.array(structure_flag, jnp.float32)*loss[0] + 1.0*binder_loss_mean
    return loss, (predicted_dict, binder_loss_mean, prob, features)

def train_step(model_params, key, batch, structure_flag):
	(loss, (predicted_dict, binder_loss, prob, features)), grads = jax.value_and_grad(get_loss_fn, has_aux=True)(model_params, key, batch, structure_flag)
	grads = norm_grads_per_example(grads, l2_norm_clip=0.1)
	grads = jax.lax.pmean(grads, axis_name='model_ax')
	loss = jax.lax.pmean(loss, axis_name='model_ax')
	return loss, grads, predicted_dict, binder_loss, prob, features

def norm_grads_per_example(grads, l2_norm_clip=0.1):
	nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
	total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
	divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
	normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
	grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
	return grads
def make_joint_msa(msa1, mtx1, msa2, mtx2):
    n1 = len(msa1)
    n2 = len(msa2)
    l1 = len(msa1[0])
    l2 = len(msa2[0])
    query_sequence = msa1[0] + msa2[0]
    query_deletion = mtx1[0] + mtx2[0]
    new_sequences1 = []
    new_sequences2 = []
    new_deletions1 = []
    new_deletions2 = []
    for i in range(n1):
        new_sequences1.append(msa1[i]+l2*'-')
        new_deletions1.append(mtx1[i] + l2*[0])
    for i in range(n2):
        new_sequences2.append(l1*'-'+msa2[i])
        new_deletions2.append(l1*[0]+mtx2[i])
    new_sequences = []
    new_deletions = []
    if new_sequences1:
        new_sequences += new_sequences1
        new_deletions += new_deletions1
    if new_sequences2:
        new_sequences += new_sequences2
        new_deletions += new_deletions2
    new_sequences.insert(0, query_sequence)
    new_deletions.insert(0, (l1+l2)*[0])
    return query_sequence, new_sequences, new_deletions

def get_template_features(
    mmcif_object: mmcif_parsing.MmcifObject,
    pdb_id: str,
    #template_sequence: str,
    #query_sequence: str,
    template_chain_id: str,
    kalign_binary_path: str):
    #original_query_sequence: str):
    ''''''

    # since we are giving exact template
    template_sequence = query_sequence = original_query_sequence = mmcif_object.chain_to_seqres[template_chain_id]
    indices = make_trivial_indices(query_sequence, 0, [])

    mapping = templates._build_query_to_hit_index_mapping(
        query_sequence, template_sequence, indices, indices,
        original_query_sequence)

    features, realign_warning = templates._extract_template_features(
        mmcif_object=mmcif_object,
        pdb_id=pdb_id,
        mapping=mapping,
        template_sequence=template_sequence,
        query_sequence=query_sequence,
        template_chain_id=template_chain_id,
        kalign_binary_path=kalign_binary_path)


    return features, realign_warning

def make_trivial_indices(sequence: str, start_index: int, indices_list: List[int]):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == '-':
            indices_list.append(-1)
        else:
            indices_list.append(counter)
        counter += 1

    return indices_list

def combine_chains(features_list, new_name):
    all_chain_features = {}
    all_chain_features['template_all_atom_positions'] = np.concatenate(
        [feature_dict['template_all_atom_positions'] for feature_dict in features_list],
        axis=0)

    all_chain_features['template_all_atom_masks'] = np.concatenate(
        [feature_dict['template_all_atom_masks'] for feature_dict in features_list],
        axis=0)

    all_chain_features['template_sequence'] = ''.encode().join(
        [feature_dict['template_sequence'] for feature_dict in features_list])

    all_chain_features['template_aatype'] = np.concatenate(
        [feature_dict['template_aatype'] for feature_dict in features_list],
        axis=0)

    all_chain_features['template_domain_names'] = [new_name.encode()]

    return (all_chain_features)

def get_mmcif(pdb_name: str, cif_base_dir: str) -> mmcif_parsing.MmcifObject:
    '''
    parses .cif file into a mmcif_parsing.MmcifObject
    '''
    cif_path = '{}/{}.cif'.format(cif_base_dir, pdb_name)
    with open(cif_path, 'r') as cif_file:
        cif_string = cif_file.read()

    parsing_result = mmcif_parsing.parse(
        file_id=pdb_name, mmcif_string=cif_string)
    #print(parsing_result)
    mmcif_object = parsing_result.mmcif_object

    return mmcif_object

def get_all_chain_template_features(mmcif_object: mmcif_parsing.MmcifObject, chain_ids: str):
    '''
    Construct template features dictionary for each chain and combine them into one
    '''
    features_list = []
    for chain_id in chain_ids:
        features, realign_warning = get_template_features(
                                        mmcif_object=mmcif_object,
                                        pdb_id=mmcif_object.file_id,
                                        template_chain_id=chain_id,
                                        kalign_binary_path='/home/amotmaen/.linuxbrew/bin/kalign')
        if realign_warning != None:
            print(realign_warning)

        features_list.append(features)

    template_features = combine_chains(features_list, new_name=mmcif_object.file_id)
    return template_features

def get_feature_dict_binder(item):
        #item = ['1k5n', 'A', 'C']
        target_crop_size = 180
        if item[-1] != 1:
                mmcif_object = get_mmcif(item[0], '/net/scratch/aditya20/ssm_mutant_cifs')
        else:
                mmcif_object = get_mmcif(item[0], '/net/scratch/aditya20/ssm_parent_cifs')                
        target_features = get_all_chain_template_features(mmcif_object, 'B')
        peptide_features = get_all_chain_template_features(mmcif_object, 'A')

        labels = np.array([float(item[-2])>0.5],np.int32)
        target_name = item[2]
        aligned_sequences, deletion_matrix = parse_a3m(open(f'{home_path}/ssm_parent_a3m/{target_name}/t000_.msa0.a3m', 'r').read())
        query_sequence_p = peptide_features['template_sequence'].decode()
        aligned_sequences_p, deletion_matrix_p = [query_sequence_p], [[0]*len(query_sequence_p)]

        query_sequence, msa, deletions = make_joint_msa(aligned_sequences_p, deletion_matrix_p, aligned_sequences, deletion_matrix)

        feature_dict = {**pipeline.make_sequence_features(sequence=query_sequence,description="none",num_res=len(query_sequence)),
                        **pipeline.make_msa_features(msas=[msa],
                       deletion_matrices=[deletions])}
        feature_dict['residue_index'][len(aligned_sequences_p[0]):] += 200

        random_seed = np.random.randint(0,999999)
        clamped = True

        with tf.device('cpu:0'):
            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed, clamped=bool(clamped))
        #print(processed_feature_dict)
        L1 = peptide_features['template_all_atom_positions'].shape[0]
        L2 = target_features['template_all_atom_positions'][:target_crop_size].shape[0]
        L = processed_feature_dict['aatype'].shape[1]
        #print(item[0],L,L1,L2)

        atom_positions = np.concatenate([peptide_features['template_all_atom_positions'], target_features['template_all_atom_positions'][:target_crop_size], np.zeros([L-L1-L2, 37, 3])], 0)
        atom_mask = np.concatenate([peptide_features['template_all_atom_masks'], target_features['template_all_atom_masks'][:target_crop_size], np.zeros([L-L1-L2, 37])], 0)
        aatype = processed_feature_dict['aatype'][0]
        L12 = L1+L2

        aatype = np.concatenate([processed_feature_dict['aatype'][0][:L12], 20*np.ones([L-L1-L2])],0).astype(np.int32)
        pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_np(aatype, atom_positions, atom_mask)
        protein_dict = {'aatype': aatype,
                        'all_atom_positions': atom_positions,
                        'all_atom_mask': atom_mask}
        protein_dict = make_atom14_positions(protein_dict)
        del protein_dict['aatype']
        for key_, value_ in protein_dict.items():
            protein_dict[key_] = np.array(value_)[None,]
        processed_feature_dict['pseudo_beta'] = np.array(pseudo_beta)[None,]
        processed_feature_dict['pseudo_beta_mask'] = np.array(pseudo_beta_mask)[None,]
        processed_feature_dict['all_atom_mask'] = np.array(atom_mask)[None,]
        processed_feature_dict['resolution'] = np.array(1.0)[None,]
        processed_feature_dict.update(protein_dict)
        rot, trans = make_transform_from_reference_np(
            n_xyz=processed_feature_dict['all_atom_positions'][0, :, 0, :],
            ca_xyz=processed_feature_dict['all_atom_positions'][0, :, 1, :],
            c_xyz=processed_feature_dict['all_atom_positions'][0, :, 2, :])
        processed_feature_dict['backbone_translation'] = trans[None,]
        processed_feature_dict['backbone_rotation'] = rot[None,]
        num_res = pseudo_beta.shape[0]
        processed_feature_dict['backbone_affine_mask'] = np.concatenate([np.ones([1,L1+L2]), np.zeros([1,L-L1-L2])], 1)
        processed_feature_dict['pdb_name'] = item[0]
        processed_feature_dict['labels'] = np.eye(2)[labels]
        processed_feature_dict['peptide_mask'] = np.concatenate([np.ones([L1]), np.zeros([L2]), np.zeros([L-L1-L2])], axis=0)
        return processed_feature_dict

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, IDs, loader, train_dict):
		self.IDs = IDs
		self.loader = loader
		self.train_dict = train_dict
	def __len__(self):
		return len(self.IDs)
	def __getitem__(self, index):
		ID = self.IDs[index]
		item = self.train_dict[ID]
		out = self.loader(item)
		return out

def collate(samples):
	#print('Collating')
	out_dict = {}
	for name in list_a:
		values = [item[name][0,] for item in samples]
		out_dict[name] = np.stack(values, axis=0)
	for name in list_b:
		values = [item[name] for item in samples] 
		out_dict[name] = np.stack(values, axis=0) 	
	aatype_ = [item['aatype'][0,] for item in samples]
	out_dict['aatype_'] = np.stack(aatype_, axis=0)
	out_dict['peptide_mask'] = [item['peptide_mask'] for item in samples]
	out_dict['labels'] = [item['labels'] for item in samples]
	out_dict['pdb_name'] = [item['pdb_name'] for item in samples]
	#print(out_dict['aatype_'].shape)  
	return out_dict

with open('/home/aditya20/experimentsWaf2/ssm_train.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    train = json.loads(json_str)

with open('/home/aditya20/experimentsWaf2/ssm_train_pos.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    train_pos = json.loads(json_str)
    
with open('/home/aditya20/experimentsWaf2/ssm_valid.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    valid = json.loads(json_str)

with open('/home/aditya20/experimentsWaf2/ssm_test.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    test = json.loads(json_str)

test_set = CustomDataset(list(test.keys()),get_feature_dict_binder,test)
train_set = CustomDataset(list(train.keys()),get_feature_dict_binder,train)
train_pos_set = CustomDataset(list(train_pos.keys()),get_feature_dict_binder,train_pos)
valid_set = CustomDataset(list(valid.keys()),get_feature_dict_binder,valid)

params_loader = {
    'shuffle': True,
    'num_workers': 8,
    'batch_size': 1,
    'collate_fn' : collate,
    'pin_memory' : False
}

test_loader = torch.utils.data.DataLoader(test_set,**params_loader)
train_loader = torch.utils.data.DataLoader(train_set,**params_loader)
train_pos_loader = torch.utils.data.DataLoader(train_pos_set,**params_loader)
valid_loader = torch.utils.data.DataLoader(valid_set,**params_loader)

scheduler = optax.linear_schedule(0.0, 1e-3, 1000, 0)

# Combining gradient transforms using `optax.chain`.
gradient_transform = optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0*0.01) #lr-coeff
)

n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)

if load_optimizer:
	opt_state = pickle.load(open(optimizer_state_path, "rb")) 
	global_step = int(np.load(global_step_path))
else:
	opt_state = gradient_transform.init(replicated_params)
	global_step = 0

loss_log = {}
for e in range(10):
        loss_log[e] = {}
        t0=time.time()
        temp_train_loss = []
        temp_lddt_ca = 0.0
        temp_weights = 0.0
        temp_distogram = []
        temp_masked_msa = []
        temp_exper_res = []
        temp_pred_lddt = []
        temp_chi_loss = []
        temp_fape = []
        temp_plddt_list = []
        temp_binder_loss_list = []
        temp_sidechain_fape = []
        print(f'EPOCH: {e+1}======>')
        for n, combined_batch in enumerate(zip(train_loader, train_pos_loader)):
            for j, batch in enumerate(combined_batch):
                structure_flag = True
                torsion_dict = atom37_to_torsion_angles(jnp.array(batch['aatype_']), jnp.array(batch['all_atom_positions']),jnp.array(batch['all_atom_mask']))
                batch['chi_mask'] = torsion_dict['torsion_angles_mask'][:,:,3:] #[B, N, 4] for 4 chi angles
                sin_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,0]
                cos_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,1]
                batch['chi_angles'] = jnp.arctan2(sin_chi, cos_chi) #[B, N, 4] for 4 chi angles
                rigidgroups_dict = atom37_to_frames(jnp.array(batch['aatype_']), jnp.array(batch['all_atom_positions']),jnp.array(batch['all_atom_mask']))
                batch.update(rigidgroups_dict)
                for key_, value_ in batch.items():
                    if key_ in list_a:
                            batch[key_] = value_[:,None,]
                    if key_ in list_c:
                            batch[key_] = value_[:,None,]
                for item_ in pdb_key_list_int:
                    batch[item_] = jnp.array(batch[item_], jnp.int32)
                batch['num_iter_recycling'] = jnp.array(np.tile(np.random.randint(0, model_config.model.num_recycle+1, 1)[None,], (batch_size, model_config.model.num_recycle)), jnp.int32)
                jax_key, subkey = random.split(jax_key)
                batch['peptide_mask'] = jnp.array(batch['peptide_mask'], dtype=jnp.float32)
                pdbname = batch['pdb_name']
                del batch['pdb_name']
                loss, grads, predicted_dict, binder_loss, prob, features = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(replicated_params, subkey, batch, structure_flag)
                plddt = np.sum(batch['backbone_affine_mask'][:,0,:]*confidence.compute_plddt(predicted_dict['predicted_lddt']['logits']))/np.sum(batch['backbone_affine_mask'][:,0,:])
                temp_plddt_list.append(plddt)
                temp_binder_loss_list.append(np.mean(binder_loss))
                if structure_flag:
                    temp_train_loss.append(np.mean(loss[0]))
                    temp_lddt_ca += np.sum(predicted_dict['predicted_lddt']['lddt_ca']*batch['backbone_affine_mask'][:,0,:])
                    temp_weights += np.sum(batch['backbone_affine_mask'][:,0,:])
                    temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
                    temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
                    temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
                    temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))
                    temp_fape.append(np.mean(predicted_dict['structure_module']['fape']))
                    temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
                global_step += 1
                #print(grads)
                updates, opt_state = gradient_transform.update(grads, opt_state)
                #print(updates.keys())
                #print('>>>>>>')
                #print(replicated_params.keys())
                replicated_params = optax.apply_updates(replicated_params, updates)
                if (global_step+1) % 1 == 0:
                    mean_loss = round(float(np.mean(temp_train_loss)),4)
                    lddt_ca = round(float(temp_lddt_ca/(temp_weights+1e-4)),4)
                    plddt = round(float(np.mean(temp_plddt_list)),4)
                    binder_loss = round(float(np.mean(temp_binder_loss_list)),4)
                    distogram = round(float(np.mean(temp_distogram)),4)
                    masked_msa = round(float(np.mean(temp_masked_msa)),4)
                    fape = round(float(np.mean(temp_fape)),4)
                    sidechain_fape = round(float(np.mean(temp_sidechain_fape)),4)
                    chi_loss = round(float(np.mean(temp_chi_loss)),4)
                    
                    print(f'Structure step: {global_step}, loss: {mean_loss}, binder: {binder_loss}, plddt: {plddt}, lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa},pdb: {pdbname}, label: {batch["labels"]}, prob1: {prob}', flush=True)
                    loss_log[e][global_step] = {'loss': mean_loss, 'binder': binder_loss, 'plddt': plddt, 'lddt': lddt_ca, 'fape': fape, 'sc_fape': sidechain_fape, 'chi': chi_loss, 'disto': distogram, 'msa': masked_msa, 'label':batch["labels"], 'probability':prob}
                    pickle.dump(loss_log, open('/net/scratch/aditya20/af2exp/model/ssm_v5_loss_log_run1.pkl', "wb"))
                    
                    temp_train_loss = []
                    temp_lddt_ca = 0.0
                    temp_weights = 0.0
                    temp_distogram = []
                    temp_masked_msa = []
                    temp_exper_res = []
                    temp_pred_lddt = []
                    temp_chi_loss = []
                    temp_fape = []
                    temp_plddt_list = []
                    temp_binder_loss_list = []
                    temp_sidechain_fape = []
                if (global_step+1) % 500 == 0:
                    np.save('/net/scratch/aditya20/af2exp/model/ssm_v5_global_step.npy', global_step)
                    save_params = jax.tree_map(lambda x: x[0,], replicated_params)
                    pickle.dump(save_params, open('/net/scratch/aditya20/af2exp/model/ssm_v5'+f'_{global_step}.pkl', "wb"))
                    pickle.dump(opt_state, open('/net/scratch/aditya20/af2exp/model/state_ssm_54'+f'_{global_step}.pkl', "wb"))
