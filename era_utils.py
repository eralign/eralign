# _*_ coding:utf-8 _*_
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torch
import scipy
import os
import time
import re
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Data() : 
    def __init__(self, data_path, mode):
        self.ent = [dict(), dict()]
        self.rel = [dict(), dict()]
        self.idx_id = {"ent":[[],[]], "rel":[[],[]]} 
        self.ent_names = dict()
        self.rel_names = dict()
        self.data_path = data_path
        self.get_keys(data_path)
        self.mode = mode

        
        #load KGs and test set
        all_triples,node_size,rel_size = self.load_triples(data_path)
        dual_triples, dual_node_size, dual_rel_size = self.triples_to_dual_triples(all_triples)
        print(data_path)
        print(f"# all triples : {len(all_triples)}\t # node : {node_size}\t # edge : {rel_size}")
        print(f"# dual triples : {len(dual_triples)}\t # dual node : {dual_node_size}\t # dual rel : {dual_rel_size}")
        
        test_pair, rel_pair = self.load_aligned_pair(data_path)
        cnt = self.get_cnt(all_triples)
        dual_cnt = self.get_cnt(dual_triples)

        adj_mat = self.get_adjacency_matrix(all_triples, node_size)
        adj_mat_dual = self.get_adjacency_matrix(dual_triples, rel_size)
        feature = self.get_feature(mode)
        feature_dual = self.get_feature(mode, dual=True)
        
        self.triples = [all_triples, dual_triples]
        self.aligned_pair = [test_pair, rel_pair]
        self.cnt = [cnt, dual_cnt]
        self.adj_mat = [adj_mat, adj_mat_dual]
        self.feature = [feature, feature_dual]

        s_ent_ids = self.idx_id["ent"][0]
        t_ent_ids = self.idx_id["ent"][1]
        s_rel_ids = self.idx_id["rel"][0]
        t_rel_ids = self.idx_id["rel"][1]
        
        print(f"# source ent : {len(s_ent_ids)}\t # target ent : {len(t_ent_ids)}")
        print(f"# source rel : {len(s_rel_ids)}\t # target rel : {len(t_rel_ids)}")
        print(f"# entity pair : {len(test_pair)}\t # rel pair : {len(rel_pair)}")
        print("------------------------------------------------------------------------------")

        
    def refine_name(self, name):
        refined = ""
        name = name.strip().replace("_"," ")
        for i in range(len(name)):
            c = name[i]
            if c.isalpha() and c.isupper() and i>0 and name[i-1].islower(): c = " "+c
            elif i>0 and name[i-1].isalpha() and c.isdecimal() : c = " "+c
            elif i>0 and name[i-1].isdecimal() and c.isalpha() : c = " "+c
            refined+=c
        return refined

    def get_keys(self, data_path):
        if "SRPRS" in data_path:
            translate = 2
        else:
            translate = 1
        with open(data_path+"/ent_ids_1", encoding="utf-8") as f :
            s_ents = f.readlines()
            s_ents = [line.split("\t") for line in s_ents]
            s_ent_ids = np.array([line[0] for line in s_ents]).astype("int64")
            for idx in range(len(s_ent_ids)):
                e_id = s_ent_ids[idx]
                self.ent[0][e_id] = idx
        self.idx_id["ent"][0] = s_ent_ids

        with open(data_path+"/ent_ids_2", encoding="utf-8") as f :
            t_ents = f.readlines()
            t_ents = [line.split("\t") for line in t_ents]
            t_ent_ids = np.array([line[0] for line in t_ents]).astype("int64")
            for idx in range(len(t_ent_ids)):
                e_id = t_ent_ids[idx]
                self.ent[1][e_id] = idx
        self.idx_id["ent"][1] = t_ent_ids

        with open(data_path+"/translated_ent.txt", encoding="utf-8") as f :
            ent_names_trans = f.readlines()
            ent_names_trans = [line.strip() for line in ent_names_trans]

        ent_names = {}
        trans_ids = s_ent_ids
        no_trans = t_ents
        if translate == 2: 
            trans_ids = t_ent_ids
            no_trans = s_ents

        for e_id, ent_name in zip(trans_ids, ent_names_trans):
            ent_names[e_id] = self.refine_name(ent_name)
        for e_id, uri in no_trans:
            ent_names[int(e_id)] = self.refine_name(uri.split("/")[-1])
        self.ent_names = ent_names

        with open(data_path+"/rel_ids_1", encoding="utf-8") as f :
            s_rels = f.readlines()
            s_rels = [line.split("\t") for line in s_rels]
            s_rel_ids = np.array([line[0] for line in s_rels]).astype("int64")
            for idx in range(len(s_rel_ids)):
                r_id = s_rel_ids[idx]
                self.rel[0][r_id] = idx
        self.idx_id["rel"][0] = s_rel_ids

        with open(data_path+"/rel_ids_2", encoding="utf-8") as f :
            t_rels = f.readlines()
            t_rels = [line.split("\t") for line in t_rels]
            t_rel_ids = np.array([line[0] for line in t_rels]).astype("int64")    
            for idx in range(len(t_rel_ids)):
                r_id = t_rel_ids[idx]
                self.rel[1][r_id] = idx
        self.idx_id["rel"][1] = t_rel_ids

        with open(data_path+"/translated_rel.txt", encoding="utf-8") as f :
            rel_names_trans = f.readlines()
            rel_names_trans = [line.strip() for line in rel_names_trans]

        rel_names = {}
        trans_ids = s_rel_ids
        no_trans = t_rels
        if translate == 2: 
            trans_ids = t_rel_ids
            no_trans = s_rels

        for r_id, rel_name in zip(trans_ids, rel_names_trans):
            rel_names[r_id] = self.refine_name(rel_name)
        for r_id, uri in no_trans:
            rel_names[int(r_id)] = self.refine_name(uri.split("/")[-1])
        self.rel_names = rel_names

    def load_triples(self, data_path):
        def reverse_triples(triples):
            reversed_triples = np.zeros_like(triples)
            for i in range(len(triples)):
                reversed_triples[i,0] = triples[i,2]
                reversed_triples[i,2] = triples[i,0]
                reversed_triples[i,1] = triples[i,1]
            return reversed_triples
        
        with open(data_path + "triples_1") as f:
            triples1 = f.readlines()
            
        with open(data_path + "triples_2") as f:
            triples2 = f.readlines()
            
        triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
        node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
        rel_size = np.max(triples[:,1]) + 1
        
        all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
        all_triples = np.unique(all_triples,axis=0)
        
        return all_triples, node_size, rel_size

    def load_aligned_pair(self, data_path):
        if "sup_pairs" not in os.listdir(data_path):
            with open(data_path + "ref_pairs") as f:
                ent_pair = f.readlines()
        else:
            with open(data_path + "ref_pairs") as f:
                ref = f.readlines()
            with open(data_path + "sup_pairs") as f:
                sup = f.readlines()
            ent_pair = ref + sup
            
        ent_pair = np.array([line.replace("\n","").split("\t") for line in ent_pair]).astype(np.int64)
        
        with open(data_path+"/rel_pairs.txt","rt", encoding="UTF8") as f:
            rel_pair = f.readlines()
            rel_pair = np.array([line.split("\t") for line in rel_pair]).astype(np.int64)

        return ent_pair, rel_pair  

    def triples_to_dual_triples(self, triples):
        def reverse_triples(triples):
            reversed_triples = np.zeros_like(triples)
            for i in range(len(triples)):
                reversed_triples[i,0] = triples[i,2]
                reversed_triples[i,2] = triples[i,0]
                reversed_triples[i,1] = triples[i,1]

            return reversed_triples

        def seperate_triples(triples):
            rel_dict = dict()

            for h,r,t in triples :

                if h not in rel_dict : rel_dict[h] = {r}
                else : rel_dict[h].add(r)

                if t not in rel_dict : rel_dict[t] = {r}
                else : rel_dict[t].add(r)
            
            return rel_dict

        rel_dict = seperate_triples(triples)

        dual_graph = []
        for ent in rel_dict.keys():
            rel_lst = list(rel_dict[ent])
            if len(rel_lst) > 1:
                for i in range(len(rel_lst)-1):
                    for j in range(i+1,len(rel_lst)):
                        dual_graph.append([rel_lst[i], ent, rel_lst[j]])
        dual_graph = np.array(dual_graph)
        dual_graph = np.concatenate([dual_graph,reverse_triples(dual_graph)],axis=0)
        dual_graph = np.unique(dual_graph,axis=0)

        node_size = max([np.max(dual_graph[:,0]),np.max(dual_graph[:,2])]) + 1
        rel_size = np.max(dual_graph[:,1]) + 1

        return dual_graph, node_size, rel_size

    def get_cnt(self, triples):
        cnt = {}
        for h,r,t in triples:
            if h in cnt:
                cnt[h].append([r,t])
            else:
                cnt[h] = [[r,t]] 
        return cnt

    def get_adjacency_matrix(self, triples, node_size):
        #build the relational adjacency matrix
        dr = {}
        for x,r,y in triples:
            if r not in dr:
                dr[r] = 0
            dr[r] += 1
            
        sparse_rel_matrix = []
        for i in range(node_size):
            sparse_rel_matrix.append([i,i,np.log(len(triples)/node_size)]);
        for h,r,t in triples:
            sparse_rel_matrix.append([h,t,np.log(len(triples)/dr[r])])

        sparse_rel_matrix = np.array(sorted(sparse_rel_matrix,key=lambda x:x[0]))
        sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:,:2],values=sparse_rel_matrix[:,2],dense_shape=(node_size,node_size))
        return sparse_rel_matrix

    def get_feature(self, mode, dual=False):
        if dual : filename = "rel"
        else: filename = "ent"
        filename = f"{filename}_emb.npy"
        word_vec = torch.tensor(np.load(self.data_path+filename)).float()
        word_vec = word_vec/((word_vec**2).sum(1)**0.5)[:,None]
        word_vec = word_vec.type(torch.DoubleTensor)
        feature = word_vec
        if mode == "hybrid" or dual:
            char_vec = self.encode_bigram(dual)
            feature = np.concatenate([word_vec, char_vec], -1)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        return feature

    def encode_bigram(self, dual):
        if dual : names = self.rel_names
        else :  names = self.ent_names

        node_size = len(names)
        #generate the bigram dictionary
        d = {}
        count = 0
        for i in range(node_size):
            name = [names[i]]
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    if word[idx:idx+2] not in d:
                        d[word[idx:idx+2]] = count
                        count += 1
        
        char_vec = np.zeros((node_size,len(d)))
        for i in range(node_size):
            name = [names[i]]
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    char_vec[i,d[word[idx:idx+2]]] += 1
                
            if np.sum(char_vec[i]) == 0:
                char_vec[i] = np.random.random(len(d))-0.5
            char_vec[i] = char_vec[i]/ np.linalg.norm(char_vec[i])

        return char_vec

class Refiner():
    def __init__(self, data, iters, tau, lam):
        self.iters = iters
        self.tau = tau
        self.lam = lam
        self.data = data
        self.ent_sims = None
        self.rel_sims = None
        self.filename = ["ent", "rel"]


    def get_filename(self, dual, mode, iters):
        if dual : filename = "rel"
        else : 
            filename = "ent"
            filename = f"{mode[0]}_{filename}"
        filename += f"{iters}"
        if iters>1:
            for i in range(iters):
                filename += f"_{int(self.lam*10):02d}"
            filename += f"t{self.tau:.0e}"
        filename += ".npy"
        return filename

    def load_file(self, dual, mode, iters):
        filename = self.get_filename(dual, mode, iters)
        file_path = self.data.data_path+"embs/"
        if filename not in os.listdir(file_path):
            return list([])
        return np.load(file_path+filename)
        
    def sinkhorn(self, sims, iters=10):
        sims1_lam = tf.exp(sims*50)

        for k in range(iters):
            sims1_lam = sims1_lam / tf.reduce_sum(sims1_lam,axis=1,keepdims=True)
            sims1_lam = sims1_lam / tf.reduce_sum(sims1_lam,axis=0,keepdims=True)
        return sims1_lam
    
    def get_init_sims(self, dual=False, depth=2):
        def cal_sims(s_ent_ids, t_ent_ids,feature):
            feature_a = tf.gather(indices=s_ent_ids , params=feature)
            feature_b = tf.gather(indices=t_ent_ids , params=feature)
            return tf.matmul(feature_a,tf.transpose(feature_b,[1,0]))
        
        data = self.data
        if dual : 
            filename = "rel"
            tag = 1
            
        else : 
            filename = "ent"
            tag = 0
        s_ids = data.idx_id[filename][0]
        t_ids = data.idx_id[filename][1]

        filename = f"{data.mode[0]}_{filename}0.npy"
        file_path = data.data_path+"embs/"

        sims = self.load_file(dual, data.mode, 0)
        
        if len(sims)==0:
            feature = data.feature[tag]
            adj_mat = data.adj_mat[tag]
            sims = cal_sims(s_ids, t_ids, feature)
            for i in range(depth):    
                feature = tf.sparse.sparse_dense_matmul(adj_mat, feature)
                feature = tf.nn.l2_normalize(feature,axis=-1)
                sims += cal_sims(s_ids, t_ids,feature)
            sims /= depth+1
            sims = self.sinkhorn(sims)
            print(f"save... {file_path+filename}")
            np.save(file_path+filename, sims)

        return sims

    def update_ent_sims(self, iters):
        
        if iters == 0 :
            print("Initialize Entity sims...")
            self.ent_sims = self.get_init_sims()
            return
        print("Updating Entity sims...")
        data = self.data
        lam = self.lam
        ent_sims = self.ent_sims
        rel_sims = self.rel_sims
        cnt = self.data.cnt[0]  

        fin_sims = self.load_file(False, data.mode, iters)
        if len(fin_sims)>0: 
            self.ent_sims = (lam)*ent_sims + (1-lam)*fin_sims
            return
        
        s_node_size, t_node_size = ent_sims.shape
        s_ent_ids = data.idx_id["ent"][0]
        t_ent_ids = data.idx_id["ent"][1]
        
        key_cnt = dict()
        for h_id in sorted(cnt.keys()):
            tag = 1
            if h_id in data.ent[0] : tag = 0
            
            for r_id, t_id in cnt[h_id]:
                r_idx = data.rel[tag][r_id]
                t_idx = data.ent[tag][t_id]

                if h_id not in key_cnt : key_cnt[h_id] = []
                key_cnt[h_id].append([r_idx,t_idx])
        
        t_cnt_idx = np.zeros(t_node_size, dtype='int32')
        idx = 0
        for t_idx in range(t_node_size):
            t_id = t_ent_ids[t_idx]
            if t_id in cnt : 
                idx += len(cnt[t_id])
            t_cnt_idx[t_idx] = idx     

        reversed_tau = 1/self.tau
        fin_sims = []

        for s_idx in tqdm(range(s_node_size)):
            s_id = s_ent_ids[s_idx]

            if s_id not in key_cnt: # check s_id has triple
                fin_sim = np.zeros(t_node_size)
                fin_sims.append(fin_sim) 
                continue;

            len_cnt_s = len(key_cnt[s_id])
            source = np.concatenate([np.repeat(key_cnt[s_id], len(key_cnt[t_id]), axis=0) if t_id in key_cnt else np.repeat(key_cnt[s_id], 0, axis=0) for t_id in t_ent_ids]).astype(int)
            target = np.vstack([np.tile(key_cnt[t_id], (len_cnt_s, 1)) if t_id in key_cnt else np.tile([None, None],(0,1)) for t_id in t_ent_ids]).astype(int)
            
            rel_sim = tf.gather_nd(
                    indices = np.column_stack([source[:,0],target[:,0]]),
                    params  = rel_sims
                    )
            rel_sim *= reversed_tau
            rel_sim = np.exp(rel_sim)

            ent_sim = tf.gather_nd(
                    indices = np.column_stack([source[:,1],target[:,1]]),
                    params  = ent_sims
                    )

            cal_mul = np.multiply(rel_sim, ent_sim)

            fin_sim = np.array([np.sum(cal_mul[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, t_node_size)])
            fin_sim = np.concatenate([[np.sum(cal_mul[0 : t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0] , fin_sim])
            
            norm = [np.sum(rel_sim[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, t_node_size)]
            norm = np.concatenate([[np.sum(rel_sim[:t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0] , norm])

            fin_sim = tf.math.divide_no_nan(fin_sim, norm)

            fin_sims.append(fin_sim)

        fin_sims = np.array(fin_sims)
        filename = self.get_filename(False, data.mode, iters)
        np.save(f"{data.data_path}/embs/{filename}", fin_sims)
        self.ent_sims = (lam)*ent_sims + (1-lam)*fin_sims
        return 

    def update_rel_sims(self, iters):
        if iters == 0 :
            print("Initialize Relation sims...")
            self.rel_sims = self.get_init_sims(dual=True)
            return
        
        print("Updating Relation sims...")
        data = self.data
        lam = self.lam
        ent_sims = self.ent_sims
        rel_sims = self.rel_sims
        dual_cnt = self.data.cnt[1]

        fin_sims = self.load_file(True, data.mode, iters)
        if len(fin_sims)>0: 
            self.rel_sims = (lam)*rel_sims + (1-lam)*fin_sims
            return 

        ent_sims = self.ent_sims
        rel_sims = self.rel_sims
        dual_cnt = self.data.cnt[1]

        s_rel_size, t_rel_size = rel_sims.shape
        s_rel_ids = data.idx_id["rel"][0]
        t_rel_ids = data.idx_id["rel"][1]

        key_dual_cnt = dict()
        for h_id in sorted(dual_cnt.keys()):
            tag = 1
            if h_id in data.rel[0] : tag = 0
            for r_id, t_id in dual_cnt[h_id]:
                r_idx = data.ent[tag][r_id]
                t_idx = data.rel[tag][t_id]
                if h_id not in key_dual_cnt : key_dual_cnt[h_id] = []
                key_dual_cnt[h_id].append([r_idx,t_idx])

        dual_t_cnt_idx = np.zeros(t_rel_size, dtype='int32')
        idx = 0
        for t_idx in range(t_rel_size):
            t_id = t_rel_ids[t_idx]
            if t_id in dual_cnt : 
                idx += len(dual_cnt[t_id])
            dual_t_cnt_idx[t_idx] = idx
        reversed_tau = 1/self.tau
        fin_sims = []
        batch_size = 512
        for s_idx in tqdm(range(s_rel_size)):
            s_id = s_rel_ids[s_idx]

            if s_id not in dual_cnt: # check s_id has triple
                fin_sim = np.zeros(t_rel_size)
                fin_sims.append(fin_sim)
                continue;
            
            len_cnt_s = len(key_dual_cnt[s_id])
            batch = batch_size//len_cnt_s + 16
            fin_sim = []
            norm = []
            fin_sim_len = 0
            # print(f"[{s_idx:04d}] ",end="")
            for epoch in range(t_rel_size//batch+1):
                start_idx = epoch*batch
                end_idx = (epoch+1)*batch
                if start_idx >= t_rel_size : break;
                if end_idx > t_rel_size : end_idx = t_rel_size

                source = np.concatenate([np.repeat(key_dual_cnt[s_id], len(key_dual_cnt[t_id]), axis=0) if t_id in key_dual_cnt else np.repeat(key_dual_cnt[s_id], 0, axis=0) for t_id in t_rel_ids[start_idx:end_idx]]).astype(int)
                target = np.vstack([np.tile(key_dual_cnt[t_id], (len_cnt_s, 1)) if t_id in key_dual_cnt else np.tile([None, None],(0,1)) for t_id in t_rel_ids[start_idx:end_idx]]).astype(int)
                
                rel_sim = tf.gather_nd(
                        indices = np.column_stack([source[:,1],target[:,1]]),
                        params  = rel_sims
                        )
                ent_sim = tf.gather_nd(
                        indices = np.column_stack([source[:,0],target[:,0]]),
                        params  = ent_sims
                        )
                ent_sim *= reversed_tau
                ent_sim = np.exp(ent_sim)
                cal_mul = np.multiply(ent_sim, rel_sim)

                if start_idx == 0 :
                    fin_sim_batch = np.array([np.sum(cal_mul[dual_t_cnt_idx[t_idx-1]*len_cnt_s : dual_t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, end_idx)])
                    fin_sim_batch = np.concatenate([[np.sum(cal_mul[0 : dual_t_cnt_idx[0]*len_cnt_s])] if dual_t_cnt_idx[0]>0 else [0], fin_sim_batch])
                    norm_batch = [np.sum(ent_sim[dual_t_cnt_idx[t_idx-1]*len_cnt_s : dual_t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, end_idx)]
                    norm_batch = np.concatenate([[np.sum(ent_sim[:dual_t_cnt_idx[0]*len_cnt_s])] if dual_t_cnt_idx[0]>0 else [0], norm_batch])
                
                else :
                    fin_sim_batch = np.array([np.sum(cal_mul[dual_t_cnt_idx[t_idx-1]*len_cnt_s-fin_sim_len : dual_t_cnt_idx[t_idx]*len_cnt_s-fin_sim_len]) for t_idx in range(start_idx, end_idx)])
                    norm_batch = np.array([np.sum(ent_sim[dual_t_cnt_idx[t_idx-1]*len_cnt_s-fin_sim_len : dual_t_cnt_idx[t_idx]*len_cnt_s-fin_sim_len]) for t_idx in range(start_idx, end_idx)])
                
                fin_sim.extend(fin_sim_batch)
                norm.extend(norm_batch)
                fin_sim_len += len(cal_mul)
                
            fin_sim = tf.math.divide_no_nan(fin_sim, norm)
            fin_sims.append(fin_sim)
            
        fin_sims = np.array(fin_sims)
        filename = self.get_filename(True, data.mode, iters)
        np.save(f"{data.data_path}/embs/{filename}", fin_sims)
        self.rel_sims = (lam)*rel_sims + (1-lam)*fin_sims
        return 

class Verifier():
    def __init__(self, refine, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.ranks = []
        self.results = []
        data = refine.data
        ent_sims2 = refine.sinkhorn(refine.ent_sims)
        rel_sims2 = refine.sinkhorn(refine.rel_sims)
        trans_ent2 = np.transpose(ent_sims2)
        trans_rel2 = np.transpose(rel_sims2)

        if refine.iters>0:
            ent_sims0 = refine.load_file(False, data.mode, 0)
            
            test_pair = data.aligned_pair[0]

            print(f"ent_sims0      > ", end="")
            ranks0, sort_sims0, results0 = eval_sims(ent_sims0, data, verify=True)
            print(f"ent_sims       > ", end="")
            ranks2, sort_sims2, results2 = eval_sims(ent_sims2, data, verify=True)
            print(f"trans_ent_sims > ", end="")
            trans_ranks2, trans_sort_sims2, trans_results2 = eval_sims(trans_ent2, data, trans=True, verify=True)
           
            consistency = self.get_consistency(ent_sims0, ent_sims2)
            confidence = self.get_confidence(sort_sims2)
            
            self.ranks = [ranks0, ranks2, trans_ranks2]
            self.results = [results0, results2, trans_results2]
            self.consistency = consistency
            self.confidence = confidence

        print(f"STS model: {model}")
        self.model_sts = SentenceTransformer(model)
        self.data = data
        self.ent_sims2 = ent_sims2
        self.rel_sims2 = rel_sims2
        self.trans_ent2 = trans_ent2
        self.trans_rel2 = trans_rel2

    def get_consistency(self, sims0, sims2):
        def get_cos_sim(A, B):
            return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
        cos_sims = []
        for sim0, sim2 in zip(sims0, sims2):
            cos_sim = get_cos_sim(sim0, sim2)
            # cos_sim = util.pytorch_cos_sim(sim0, sim2)[0][0]
            cos_sims.append(cos_sim)
        cos_sims = np.array(cos_sims)
        consistency = cos_sims ** 3
        return consistency

    def get_confidence(self, sort_sims2):
        return sort_sims2[:,0]-sort_sims2[:,1]

    def get_premise_hypothesis(self, s_idx, topk=20, check=False, trans=False, sentencek=10):
        data = self.data
        
        if trans: 
            ranks2 = self.ranks[2]
            ent_sims, rel_sims = self.trans_ent2, self.trans_rel2
            tag = (1,0)
        else: 
            ent_sims, rel_sims = self.ent_sims2, self.rel_sims2
            ranks2 = self.ranks[1]
            tag = (0,1)
        ent_names, rel_names = data.ent_names, data.rel_names
        cnt = data.cnt[0]
        rank_list = ranks2[s_idx,:topk]


        s_id = data.idx_id["ent"][tag[0]][s_idx]
        premise = ent_names[s_id]

        target_ids = data.idx_id["ent"][tag[1]]
        target_rank_ids = np.take(target_ids, rank_list)

        premise_idx_list = []
        s_tail_cnt = np.zeros(len(cnt[s_id]))
        for i in range(len(cnt[s_id])):
            r_id, t_id = cnt[s_id][i] 
            r_idx = data.rel[tag[0]][r_id]
            t_idx = data.ent[tag[0]][t_id]
            premise_idx_list.append([r_idx, t_idx])
            if t_id in cnt : s_tail_cnt[i] += len(cnt[t_id])
        s_tail_cnt = np.argsort(-s_tail_cnt)[:sentencek] # 자주 나오는 것 우선
        premise_idx_list = np.take(premise_idx_list, s_tail_cnt, axis=0)

        premise += ", which"

        for i in s_tail_cnt:
            r_id, t_id = cnt[s_id][i]
            premise += f" {rel_names[r_id]} is {ent_names[t_id]},"

        premise = premise[:-1] + "."


        hypothesis_list = []
        for t_head_id in target_rank_ids :
            hypothesis = ent_names[t_head_id]
            hypothesis += ", which"
            
            hypo_idx_list = []
            target_rel_dict = {}
            for r_id, t_id in cnt[t_head_id]:
                r_idx = data.rel[tag[1]][r_id]
                t_idx = data.ent[tag[1]][t_id]
                hypo_idx_list.append([r_idx, t_idx])
                if r_idx not in target_rel_dict: target_rel_dict[r_idx] = set()
                target_rel_dict[r_idx].add(t_idx)
            hypo_idx_list = np.array(hypo_idx_list)

            for sr_idx, st_idx in premise_idx_list:
                hypo_rel_sims = np.take(rel_sims[sr_idx], hypo_idx_list[:, 0])
                hypo_rel_ranks = np.take(hypo_idx_list[:, 0] ,np.argsort(hypo_rel_sims))
                best_rel_idx = hypo_rel_ranks[-1]
                best_rel_id = data.idx_id["rel"][tag[1]][best_rel_idx]
        
                tail_list = sorted(list(target_rel_dict[best_rel_idx]))
                target_tail_sims = np.take(ent_sims[st_idx], tail_list)
                target_tail_ranks = np.take(tail_list, np.argsort(target_tail_sims))
                best_tail_idx = target_tail_ranks[-1]
                best_tail_id = data.idx_id["ent"][tag[1]][best_tail_idx]
                hypothesis += f" {rel_names[best_rel_id]} is {ent_names[best_tail_id]},"
            
            hypothesis = hypothesis[:-1] + "."

            hypothesis_list.append(hypothesis)
        
        return premise, hypothesis_list

    def verify(self, p1=20, p2=20, topk=20, sentencek=10):
        model_sts = self.model_sts
        data = self.data
        ranks2   = self.ranks[1]
        results2 = self.results[1]
        trans_ranks2 = self.ranks[2]
        confidence = self.confidence
        consistency = self.consistency
        test_pair = data.aligned_pair[0]
        triples = data.triples[0]
        cnt = data.cnt[0]

        confidence_threshold = np.percentile(confidence,p1)
        consistency_threshold = np.percentile(consistency,p2)
        print(f"Threshold: confidence-{p1}%({confidence_threshold:.4f}), consistency-{p2}%({consistency_threshold:.4f})")
        verify_ranks = []
        verified_idx = []
        append_count = 0
        reject_count = 0 

        for i in tqdm(range(len(test_pair))):
            s_id = test_pair[i,0]
            s_idx = data.ent[0][s_id]
            rank_list = ranks2[s_idx,:topk]
            if results2[i][1]>topk-1 :
                verify_ranks.append(rank_list)
                continue
            
            if (confidence[s_idx]>confidence_threshold and consistency[s_idx]>consistency_threshold) :
                verify_ranks.append(rank_list)
                continue

            s_premise, s_hypothesis_list = self.get_premise_hypothesis(s_idx, topk=topk, sentencek=sentencek)
            s_rank = self.sts(s_premise, s_hypothesis_list)
            top1_t_idx=ranks2[s_idx, s_rank[0]]
            t_premise, t_hypothesis_list = self.get_premise_hypothesis(top1_t_idx, topk=topk, trans=True, sentencek=sentencek)
            t_rank = self.sts(t_premise, t_hypothesis_list)
            top1_s_idx=trans_ranks2[top1_t_idx, t_rank[0]]
            
            if top1_s_idx != i :
                verify_ranks.append(rank_list)
                reject_count += 1
                continue
            s_rank = np.take(a=rank_list, indices=s_rank)
            verify_ranks.append(s_rank)
            append_count += 1
            verified_idx.append(i)
        verify_ranks = np.array(verify_ranks)
        print(f"append count: {append_count}")
        print(f"reject count: {reject_count}")

        return verify_ranks, verified_idx
    
    def sts(self, premise, hypothesis_list) :
        model=self.model_sts
        premise_emb = model.encode(premise)
        hypo_sim_list = []
        for hypo in hypothesis_list:
            hypo_sim_list.append(util.pytorch_cos_sim(premise_emb, model.encode(hypo))[0][0])
            hypo_emb = model.encode(hypo)
        # print(hypo_sim_list)
        sim_rank = np.argsort(np.array(hypo_sim_list)*-1)
        return sim_rank

    def verified_dist(self, new_results, verified_idx, show_all=False):
        results = self.results[1]
        verified_case_keys = ["o->o", "o->x", "x->o", "x->x"]
        verified_case = dict()
        for key in verified_case_keys:
            verified_case[key] = []

        changed=[0,0,0]
        for test_pair_idx in verified_idx:
            before = results[test_pair_idx][1]
            after = new_results[test_pair_idx][1]
            if before-after > 0 :changed[0] += 1
            elif before-after <0 : changed[-1] +=1
            else : changed[1]+= 1

            item = [test_pair_idx, int(before), int(after)]
            if before>0:
                if after>0: verified_case[verified_case_keys[3]].append(item)
                else: verified_case[verified_case_keys[2]].append(item)
            else : 
                if after>0: verified_case[verified_case_keys[1]].append(item)
                else: verified_case[verified_case_keys[0]].append(item)
        print(f"better : {changed[0]}, worse : {changed[-1]}, non-changed : {changed[1]}")
        for i in verified_case_keys:
            print(f"{i} : {len(verified_case[i])} ")
        print()
        if show_all:
            for i in verified_case_keys:
                print(f"{i} : {len(verified_case[i])}")
                for case in verified_case[i]:
                    print(case)
                print("===============================")

        return verified_case

    def new_rank_results (self, verify_ranks, batch_size=1024):
        ranks = self.ranks[1]
        data = self.data
        test_pair = data.aligned_pair[0]
        ent_sims = self.ent_sims2
        results = self.results[1]

        def cal(results):
            hits1,hits10,mrr = 0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits10,mrr

        ans = {}
        ans_keys = []
        for s_id, t_id in test_pair: 
            s_idx = data.ent[0][s_id]
            t_idx = data.ent[1][t_id]
            ans[s_idx] = t_idx
            ans_keys.append(s_idx)


        new_ranks = deepcopy(ranks)
        verify_ranks = verify_ranks.astype(int)
        for i in range(len(ans_keys)):
            s_idx = ans_keys[i]
            new_ranks[s_idx] =  np.concatenate([verify_ranks[i], new_ranks[s_idx, len(verify_ranks[0]):]])
        new_ranks = np.array(new_ranks)


        new_results = []
        for epoch in range(len(test_pair)//batch_size+1) :
            rank = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=new_ranks)
            ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
            new_results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(ent_sims[0])]))).numpy())
        new_results = np.concatenate(new_results,axis=0)

        hits1,hits10,mrr = cal(new_results)
        print(f"hits@1 : {hits1/len(new_results)*100 :.2f} hits@10 : {hits10/len(new_results)*100 :.2f} MRR : {mrr/len(new_results)*100 :.2f}")
        # print(len(new_results))
        
        return new_ranks, new_results


def eval_sims(sims, data, trans=False, verify=False, batch_size = 1024):
    def cal(results):
        hits1,hits10,mrr = 0,0,0
        for x in results[:,1]:
            if x < 1:
                hits1 += 1
            if x < 10:
                hits10 += 1
            mrr += 1/(x + 1)
        return hits1,hits10,mrr
        
    ans = {}
    ans_keys = []
    test_pair = data.aligned_pair[0]
    for s_id, t_id in test_pair: 
        if trans:
            ans[data.ent[1][t_id]] = data.ent[0][s_id]
            ans_keys.append(data.ent[1][t_id])
        else:
            ans[data.ent[0][s_id]] = data.ent[1][t_id]
            ans_keys.append(data.ent[0][s_id])

    results = []
    ranks = []
    sort_sims = []
    if verify:
        for epoch in range(len(sims)//batch_size+1) :
            sim = sims[epoch*batch_size : (epoch+1)*batch_size]
            rank = tf.argsort(-sim,axis=-1)        
            ranks.append(rank)
        ranks = np.concatenate(ranks,axis=0)
        for i in range(len(sims)):
            sort_sims.append(np.take(sims[i], ranks[i]))
        sort_sims = np.array(sort_sims)

        for epoch in range(len(test_pair)//batch_size+1) :
            test_pair_rank = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=ranks)
            ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
            results.append(tf.where(tf.equal(tf.cast(test_pair_rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(sims[0])]))).numpy())
        
    else:
        for epoch in range(len(test_pair)//batch_size+1) :
            sim = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=sims)
            test_pair_rank = tf.argsort(-sim,axis=-1)
            ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
            results.append(tf.where(tf.equal(tf.cast(test_pair_rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(sims[0])]))).numpy())
    
    results = np.concatenate(results,axis=0)
    hits1,hits10,mrr = cal(results)
    print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(results)*100,hits10/len(results)*100,mrr/len(results)*100))
    return ranks, sort_sims, results

    
