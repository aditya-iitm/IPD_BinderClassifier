import os
import gzip
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import string
import optax
import csv
import collections
import argparse
import json
from dateutil import parser
from itertools import combinations
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
from typing import Any, Mapping, Optional, Sequence, Tuple
from alphafold.common.residue_constants import atom_types, restype_name_to_atom14_names
import pandas as pd
from train_utils import *
from typing import Tuple
import functools
import time
import pickle
import torch
from jax import random
from jax import jit
import tensorflow.compat.v1 as tf1
warnings.filterwarnings("ignore")
flags = tf1.app.flags

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
        logits1 = jax.nn.gelu(hk.Linear(8)(features_binned))
        binder_logits = hk.Linear(2)(logits1)



        return binder_logits, features
    
    

def binder_classification_fn(input_features, training):
    model = BinderClassifier(0.1)(
        input_features,
        training=training
    )
    return model

rng = jax.random.PRNGKey(43)
binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

batch_size = 1

### CUSTOM DATASET CLASS ###

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
    
### MODEL CONFIG ####

load_optimizer = True
optimizer_state_path = '/net/scratch/aditya20/af2exp/model/new_binder_v1_299_state.pkl'
global_step_path     = '/net/scratch/aditya20/af2exp/model/af2_binder_v1_global_step.npy'
model_weights_path   = '/net/scratch/aditya20/af2exp/model/new_binder_v1_299.pkl'

jax_key = jax.random.PRNGKey(0)
model_name = "model_3_ptm" #no-templates
model_config = config.model_config(model_name)
model_config.data.common.resample_msa_in_recycling = True
model_config.model.resample_msa_in_recycling = True
model_config.data.common.max_extra_msa = 512
model_config.data.eval.max_msa_clusters = 512
model_config.data.eval.crop_size = 256
model_config.model.heads.structure_module.structural_violation_loss_weight = 1.0
model_config.model.embeddings_and_evoformer.evoformer_num_block = 48

if load_optimizer:
    full_model_params = pickle.load(open(model_weights_path, "rb"))
    binder_model_params, af2_model_params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", full_model_params)
else:
    af2_model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/projects/ml/alphafold/")
    binder_model_params = pickle.load(open('/net/scratch/aditya20/af2exp/model/binder_params_binned.pkl', "rb"))
    #binder_model_params = pickle.load(open('/net/scratch/aditya20/af2exp/model/binder_params.pkl', "rb"))
    #binder_model_params = pickle.load(open('/home/justas/projects/lab_github/alphafold-class/models/binder_params.pkl', "rb"))    
model_runner = model.RunModel(model_config, af2_model_params)
model_params = hk.data_structures.merge(af2_model_params, binder_model_params)

### REQUIRED FUNCTIONS ###

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

def seq2np(seq):
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.uint8)
    seq = np.array([list(s) for s in seq], dtype='|S1').view(np.uint8)
    for i,a in enumerate(abc):
        seq[seq == a] = i
    #mask = ((seq<0)+(seq>19)).sum(-1)==0
    #seq = seq[mask]
    return seq.astype(int).flatten()

### INITIALIZE OTHER VARIABLES ###

params = {
    "LIST"    : "/projects/ml/TrRosetta/PDB-20212AUG02/list_v00.csv",
    "VAL"     : "/projects/ml/TrRosetta/PDB-20212AUG02/val/xaa",
    "DIR"     : "/projects/ml/TrRosetta/PDB-2021AUG02",
    "DATCUT"  : "2030-Jan-01",
    "RESCUT"  : 2.5,
    "HOMO"    : 0.90 # min seq.id. to detect homo chains
}

params_loader = {
    'shuffle': True,
    'num_workers': 0,
    'batch_size': 1,
    'collate_fn' : collate,
    'pin_memory' : False
}

RES_NAMES = [
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
]

RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYV'

to1letter = {aaa:a for a,aaa in zip(RES_NAMES_1,RES_NAMES)}
to3letter = {a:aaa for a,aaa in zip(RES_NAMES_1,RES_NAMES)}

ATOM_NAMES = [
    ("N", "CA", "C", "O", "CB"), # ala
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), # arg
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"), # asn
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"), # asp
    ("N", "CA", "C", "O", "CB", "SG"), # cys
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"), # gln
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"), # glu
    ("N", "CA", "C", "O"), # gly
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"), # his
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"), # ile
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"), # leu
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"), # lys
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"), # met
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), # phe
    ("N", "CA", "C", "O", "CB", "CG", "CD"), # pro
    ("N", "CA", "C", "O", "CB", "OG"), # ser
    ("N", "CA", "C", "O", "CB", "OG1", "CG2"), # thr
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"), # trp
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), # tyr
    ("N", "CA", "C", "O", "CB", "CG1", "CG2"), # val
]

id_hash = {}
with open('PDB-2021AUG02/list_v01.csv') as g:
    for line in g:
        words = line.strip().split(',')
        id_hash[words[0]] = words[3]
        
a2i = {a:i for i,a in enumerate(atom_types)}
t14_to_t37 = []
for i,aas in enumerate(ATOM_NAMES):
    t14_to_t37.append(np.array([[j,a2i[a]] for j,a in enumerate(aas)]))
    
### FUNCTION TO ASSEMBLE XYZ COORDS FROM PYTORCH FILE ###

def assemble_from_pt(line):
    words = line.strip().split(' ')
    pdbid = words[0]
    chain1 = words[2]
    xform_idx10 = words[3]
    xform_idx11 = words[4]
    chain2 = words[5]
    xform_idx20 = words[6]
    xform_idx21 = words[7]
    hash1 = pdbid+'_'+chain1
    hash2 = pdbid+'_'+chain2

    PREFIX = "%s/torch/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)

    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])
    chains = set([chain1,chain2])

    # load relevant chains
    chains_meta = {c:torch.load("%s_%s.pt"%(PREFIX,c)) for c in chains}

    # generate assembly
    asmb = {}
    xform1 = meta['asmb_xform%d'%int(xform_idx10)]
    u1 = xform1[:,:3,:3]
    r1 = xform1[:,:3,3]  

    xform2 = meta['asmb_xform%d'%int(xform_idx20)]
    u2 = xform2[:,:3,:3]
    r2 = xform2[:,:3,3]  

    # transform selected chains 
    xyz = chains_meta[chain1]['xyz']
    xyz_ru = torch.einsum('bij,raj->brai', u1, xyz) + r1[:,None,None,:]
    asmb.update({(chain1):xyz_i for i,xyz_i in enumerate(xyz_ru)})

    xyz = chains_meta[chain2]['xyz']
    xyz_ru = torch.einsum('bij,raj->brai', u2, xyz) + r2[:,None,None,:]
    asmb.update({(chain2):xyz_i for i,xyz_i in enumerate(xyz_ru)})

    seq,xyz,idx,masked = "",[],[],[]
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains_meta[k]['seq']
        xyz.append(v)
        idx.append(torch.full((v.shape[0],),counter))
        masked.append(chains_meta[k]['mask'])
    
    start1 = np.random.randint(len(chains_meta[chain1]['seq'])-128+1)
    start2 = np.random.randint(len(chains_meta[chain2]['seq'])-128+1)
    return {
            chain1:
            {'seq':chains_meta[chain1]['seq'] [start1:start1+128],
             'xyz':asmb[chain1][start1:start1+128,:,:],
             'idx':torch.full((asmb[chain1].shape[0],),0)[start1:start1+128],
             'mask':chains_meta[chain1]['mask'][start1:start1+128,:],
             'start1':start1,
             'msa_hash':id_hash[hash1]},
            chain2:
             {'seq':chains_meta[chain2]['seq'] [start2:start2+128],
              'xyz':asmb[chain2][start2:start2+128,:,:],
              'idx':torch.full((asmb[chain2].shape[0],),1)[start2:start2+128],
              'mask':chains_meta[chain2]['mask'][start2:start2+128,:],
              'start2':start2,
              'msa_hash':id_hash[hash2]}
            }

### FUNCTIONS TO ASSEMBLE XYZ COORDS FROM PDB FILE (INDENTATION IS DIFFERENT) ###
def get_seq_from_pdb( pdb_fn ):
  to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

  seq = []
  start = 0
  seqstr = ''

  with open(pdb_fn) as fp:
    for line in fp:

      if not line.startswith("ATOM"):
        continue
      if line[12:16].strip() != "CA":
        continue
      #print(int(line[22:27].strip()))
      resName = line[17:20]
      
      if int(line[22:27].strip()) == start+1:
        start += 1
        seqstr += to1letter[resName]
        continue
      else:
        seq.append(seqstr)
        seqstr = ''
        seqstr += to1letter[resName]
        start += 201
  seq.append(seqstr)
  return seq

def af2_get_atom_positions( pdbfilename ) -> Tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""

  with open(pdbfilename, 'r') as pdb_file:
    lines = pdb_file.readlines()

  # indices of residues observed in the structure
  idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
  num_res = len(idx_s)

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)

  residues = collections.defaultdict(list)
  # 4 BB + up to 10 SC atoms
  xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
  for l in lines:
    if l[:4] != "ATOM":
        continue
    resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]

    residues[ resNo ].append( ( atom.strip(), aa, [float(l[30:38]), float(l[38:46]), float(l[46:54])] ) )

  for resNo in residues:

    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)

    for atom in residues[ resNo ]:
      atom_name = atom[0]
      x, y, z = atom[2]
      if atom_name in residue_constants.atom_order.keys():
        pos[residue_constants.atom_order[atom_name]] = [x, y, z]
        mask[residue_constants.atom_order[atom_name]] = 1.0
      elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
        # Put the coordinates of the selenium atom in the sulphur column.
        pos[residue_constants.atom_order['SD']] = [x, y, z]
        mask[residue_constants.atom_order['SD']] = 1.0

    idx = idx_s.index(resNo) # This is the order they show up in the pdb
    all_positions[idx] = pos
    all_positions_mask[idx] = mask
  # _check_residue_distances(
  #     all_positions, all_positions_mask, max_ca_ca_distance) # AF2 checks this but if we want to allow massive truncations we don't want to check this

  return all_positions, all_positions_mask

def af2_all_atom_from_struct( pdbfilename, just_target=False ):
  seq_list = get_seq_from_pdb(pdbfilename)
  #print(seq_list)
  template_seq = ''.join( seq_list )

  # Parse a residue mask from the chainbreak sequence
  binder_len = len( seq_list[1] )
  residue_mask = [ int( i ) > binder_len for i in range( 1, len( template_seq ) + 1 ) ]

  all_atom_positions, all_atom_mask = af2_get_atom_positions( pdbfilename )
  #print(residue_mask)
  all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

  templates_all_atom_positions = []

  # Initially fill will all zero values
  for _ in template_seq:
    templates_all_atom_positions.append(
        jnp.zeros((residue_constants.atom_type_num, 3)))

  for idx, i in enumerate( template_seq ):
    if just_target and not residue_mask[ idx ]: continue

    templates_all_atom_positions[ idx ] = all_atom_positions[ idx ][0] # assign target indices to template coordinates

  return jnp.array(templates_all_atom_positions),jnp.array(all_atom_mask)

### GET FEATURE DICT ### 

def get_pos_feature_dict(entry):
    
    words = entry[0].strip().split(' ')
    pdbname = words[0]+'_'+words[2]+words[5]
    
    assembly = assemble_from_pt(entry[0])
    
    target_feat = assembly[list(assembly.keys())[0]]
    binder_feat = assembly[list(assembly.keys())[1]]
    start1 = target_feat['start1']
    start2 = binder_feat['start2']
    hash1 = target_feat['msa_hash']
    hash2 = binder_feat['msa_hash']
    
    target_aligned_sequences, target_deletion_matrix = parse_a3m(gzip.open(f'/home/aditya20/experimentsWaf2/PDB-2021AUG02/a3m/{hash1[:3]}/{hash1}.a3m.gz', 'rt').read())
    binder_aligned_sequences, binder_deletion_matrix = parse_a3m(gzip.open(f'/home/aditya20/experimentsWaf2/PDB-2021AUG02/a3m/{hash2[:3]}/{hash2}.a3m.gz', 'rt').read())
    
    target_sequences = [line[start1:start1+128]for line in target_aligned_sequences]
    target_matrix    = [line[start1:start1+128]for line in target_deletion_matrix]
    binder_sequences = [line[start2:start2+128]for line in binder_aligned_sequences]
    binder_matrix    = [line[start2:start2+128]for line in binder_deletion_matrix]
    
    #print(target_sequences[0],target_matrix[0])
    query_sequence, msa, deletions = make_joint_msa(binder_sequences, binder_matrix, target_sequences, target_matrix)
    
    #print(query_sequence)
    p_accept = 1/512*np.maximum(np.minimum(len(query_sequence), 512), 256)
    
    feature_dict = {**pipeline.make_sequence_features(sequence=query_sequence,description="none",num_res=len(query_sequence)),
                    **pipeline.make_msa_features(msas=[msa],
                   deletion_matrices=[deletions])}
    feature_dict['residue_index'][len(binder_aligned_sequences[0]):] += 200

    random_seed = np.random.randint(0,999999)
    clamped = True
    
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed, clamped=bool(clamped))
    
    aatype = seq2np(query_sequence)
    L = aatype.shape[0]
    xyz  = torch.cat((binder_feat['xyz'],target_feat['xyz']),axis=0).numpy()
    mask = torch.cat((binder_feat['mask'],target_feat['mask']),axis=0).numpy()
    
    xyz37 = np.zeros((L,37,3))
    mask37 = np.zeros((L,37),dtype=bool)
    for i in range(aatype.shape[0]):
            ri = aatype[i]
            if ri>=19:
                    continue
            i14 = t14_to_t37[ri][:,0]
            i37 = t14_to_t37[ri][:,1]
            xyz37[i][i37] = xyz[i][i14]
            mask37[i][i37] = mask[i][i14]

    xyz37[np.isnan(xyz37)] = 0.0
    
    #print(xyz37.shape,mask37.shape)
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, xyz37, mask37)
    start_idx = np.array(processed_feature_dict['num_res_crop_start'])[0]
    crop_size = model_config.data.eval.crop_size
    seq_length= np.array(processed_feature_dict['seq_length'])[0]
    #idx = protein.residue_index
    L = len(query_sequence)
    idx = np.arange(L)
    protein_dict = {'aatype': aatype,
            'all_atom_positions': xyz37,
            'all_atom_mask': mask37}
    #print(protein_dict['aatype'].shape, protein_dict['atom_positions'].shape, protein_dict['atom_mask'].shape)
    protein_dict = make_atom14_positions(protein_dict)
    del protein_dict['aatype']
    for key_, value_ in protein_dict.items():
            protein_dict[key_] = np.array(value_)[None,]
    processed_feature_dict['pseudo_beta'] = np.array(pseudo_beta)[None,]
    processed_feature_dict['pseudo_beta_mask'] = np.array(pseudo_beta_mask)[None,]
    processed_feature_dict['all_atom_mask'] = mask37[None,] #np.array(protein.atom_mask)[None,]
    processed_feature_dict['resolution'] = np.array(1.0)[None,]
    processed_feature_dict.update(protein_dict)

    rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=processed_feature_dict['all_atom_positions'][0, :, 0, :],
            ca_xyz=processed_feature_dict['all_atom_positions'][0, :, 1, :],
            c_xyz=processed_feature_dict['all_atom_positions'][0, :, 2, :])
    
    processed_feature_dict['backbone_translation'] = trans[None,]
    processed_feature_dict['backbone_rotation'] = rot[None,]
    num_res = pseudo_beta.shape[0]
    processed_feature_dict['backbone_affine_mask'] = np.ones([1,num_res])
    processed_feature_dict['pdb_name'] = pdbname
    label = np.array([1],np.int32)
    processed_feature_dict['labels'] = np.eye(2)[label]
    processed_feature_dict['peptide_mask'] = np.concatenate([np.ones([128]), np.zeros([128]), np.zeros([L-256])], axis=0)
    '''
    for key_, value_ in processed_feature_dict.items():

                shape_1 = np.array(processed_feature_dict[key_].shape)
                print(key_,shape_1)
                shape_1[1] = L #make zeros of size [B, L,...]
                zeros = np.zeros(shape_1)
                zeros[:,idx,] = processed_feature_dict[key_]
                if L >= crop_size:
                        b = zeros[:,start_idx:start_idx+crop_size,]
                else:
                        shape_1 = np.array(processed_feature_dict[key_].shape)
                        shape_1[1] = crop_size
                        b = np.zeros(shape_1)
                        b[:,start_idx:start_idx+L,] = zeros
                processed_feature_dict[key_] = b
    '''
    return processed_feature_dict   

def get_neg_feature_dict(entry):
    
    target = entry[1]
    binder = entry[0]
    
    pdbfile = f'/home/aivan/for/Aditya/af2_negative_set/single/{target}/{binder}_unrelaxed_model_1.pdb'
    
    all_atom,all_mask = af2_all_atom_from_struct(pdbfile)
    
    target_sequences, target_matrix = parse_a3m(open(f'/home/aivan/for/Aditya/af2_negative_set/targets/{target}.a3m').read())
    binder_sequences, binder_matrix = [binder],[[0]*len(binder)]
    query_sequence, msa, deletions = make_joint_msa(target_sequences, target_matrix, binder_sequences, binder_matrix)
    #print(query_sequence)
    
    feature_dict = {**pipeline.make_sequence_features(sequence=query_sequence,description="none",num_res=len(query_sequence)),
                    **pipeline.make_msa_features(msas=[msa],
                   deletion_matrices=[deletions])}
    feature_dict['residue_index'][len(target_sequences[0]):] += 200

    random_seed = np.random.randint(0,999999)
    clamped = True
    
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed, clamped=bool(clamped))
    
    aatype = seq2np(query_sequence)
    L = processed_feature_dict['aatype'].shape[1]
    L1 = len(target_sequences[0])
    L2 = len(binder)
    L12 = L1+L2
    #print(L,L1,L2,aatype.shape[0])
    
    aatype = np.concatenate([processed_feature_dict['aatype'][0][:L12], 20*np.ones([L-L1-L2])],0).astype(np.int32)
    atom_positions = np.concatenate([all_atom, np.zeros([L-L1-L2, 37, 3])], 0)
    atom_mask = np.concatenate([all_mask, np.zeros([L-L1-L2, 37])], 0)
    
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, atom_positions, atom_mask)
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
    
    rot, trans = quat_affine.make_transform_from_reference(
        n_xyz=processed_feature_dict['all_atom_positions'][0, :, 0, :],
        ca_xyz=processed_feature_dict['all_atom_positions'][0, :, 1, :],
        c_xyz=processed_feature_dict['all_atom_positions'][0, :, 2, :])
    
    processed_feature_dict['backbone_translation'] = trans[None,]
    processed_feature_dict['backbone_rotation'] = rot[None,]
    num_res = pseudo_beta.shape[0]
    processed_feature_dict['backbone_affine_mask'] = np.concatenate([np.ones([1,L1+L2]), np.zeros([1,L-L1-L2])], 1)
    processed_feature_dict['pdb_name'] = target+'_'+binder
    label = np.array([0],np.int32)
    processed_feature_dict['labels'] = np.eye(2)[label]
    processed_feature_dict['peptide_mask'] = np.concatenate([np.zeros([L1]), np.ones([L2]), np.zeros([L-L1-L2])], axis=0)
    
    return processed_feature_dict

    
with open('/home/aditya20/experimentsWaf2/binder_train_pos1.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    train_pos = json.loads(json_str)

with open('/home/aditya20/experimentsWaf2/binder_train_neg.json', 'r') as json_file:
        json_list = list(json_file)
for json_str in json_list:
    train_neg = json.loads(json_str)
    
positive_set = CustomDataset(list(train_pos.keys()),get_pos_feature_dict,train_pos)
negative_set = CustomDataset(list(train_neg.keys()),get_neg_feature_dict,train_neg)

loader1 = torch.utils.data.DataLoader(positive_set,**params_loader)
loader2 = torch.utils.data.DataLoader(negative_set,**params_loader)

scheduler = optax.linear_schedule(0.0, 1e-3, 200, 0)

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

#print(replicated_params)
for e in range(10):
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
        for n, combined_batch in enumerate(zip(loader1, loader2)):
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
                pdb_name = batch['pdb_name']
                del batch['pdb_name']   #IMPORTANT, PMAP DOESNT WORK IF this isnt deleted, because of it not having a shape
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
                updates, opt_state = gradient_transform.update(grads, opt_state)
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
                    print(f'Structure step: {global_step}, loss: {mean_loss}, binder: {binder_loss}, plddt: {plddt}, lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa},pdb: {pdb_name}, label: {batch["labels"]}, prob1: {prob}', flush=True)
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
                if (global_step+1) % 100 == 0:
                    np.save('/net/scratch/aditya20/af2exp/model/af2_binder_v1_global_step.npy', global_step)
                    save_params = jax.tree_map(lambda x: x[0,], replicated_params)
                    pickle.dump(save_params, open('/net/scratch/aditya20/af2exp/model/new_binder_v1'+f'_{global_step}.pkl', "wb"))
                    pickle.dump(opt_state, open('/net/scratch/aditya20/af2exp/model/new_binder_v1'+f'_{global_step}_state.pkl', "wb"))

