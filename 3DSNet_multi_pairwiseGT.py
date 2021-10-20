import os
import re
import time
import numpy as np
import random
import glob
import matplotlib
import matplotlib.pyplot as plt
import statistics
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

import scipy.ndimage

from skimage import exposure

from sklearn.metrics import roc_curve

import csv
from itertools import zip_longest

import sklearn


torch.backends.cudnn.enabled = False
torch.manual_seed(42)
print(torch.random.initial_seed())

def plotExampleImage(image,title):
	fig = plt.figure(figsize=(10,2))
	plt.title(title)
	cols = 3
	rows = 1
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	proj1 = np.mean(volume, axis=1)
	proj2 = np.mean(volume, axis=2)
	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.title.set_text("axis 0")
	plt.imshow(proj0,cmap="gray") 
	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.title.set_text("axis 1")
	plt.imshow(proj1,cmap="gray")
	ax3 = fig.add_subplot(rows, cols, 3)
	ax3.title.set_text("axis 2")
	plt.imshow(proj2,cmap="gray")

def extractInfos(f):
	dict_visit = {"bl":0,"m06":1,"m12":2,"m24":3}
	dict_modality = {"clin":0,"mr":1}
	try:
		regex = re.match(r"(Dementia|CN|MCI)_(clin|mr)(bl|m06|m12|m24)_(\d{3}_S_\d{4})_L([0|1]).pt",f)
		assert regex != None
		diag_id = regex.group(1)
	except AssertionError:
		f = "N/A"+f
		regex = re.match(r"(N\/A)_(clin|mr)(bl|m06|m12|m24)_(\d{3}_S_\d{4})_L([0|1]).pt",f)
		diag_id = ""
	modality_id = regex.group(2)
	visit_id = regex.group(3)
	subject_id = regex.group(4)
	
	infos = (subject_id, dict_visit[visit_id], dict_modality[modality_id])
	return infos  
	
def extractInfos2(f):
	dict_visit = {"bl":0,"m06":1,"m12":2,"m24":3}
	dict_modality = {"clin":0,"mr":1}
	try:
		regex = re.match(r"(Dementia|CN|MCI)_(clin|mr)(bl|m06|m12|m24)_(\d{3}_S_\d{4})_(L[0|1]).pt",f)
		assert regex != None
		diag_id = regex.group(1)
	except AssertionError:
		f = "N/A"+f
		regex = re.match(r"(N\/A)_(clin|mr)(bl|m06|m12|m24)_(\d{3}_S_\d{4})_(L[0|1]).pt",f)
		diag_id = ""
	modality_id = regex.group(2)
	visit_id = regex.group(3)
	subject_id = regex.group(4)
	class_id = regex.group(5)
	infos = [diag_id, dict_visit[visit_id], dict_modality[modality_id],subject_id, class_id]
	return infos
	
def mmse_group1(mmse):
	if mmse == None:
		return None
	if mmse <= 10:
		group = 4
	elif mmse >= 11 and mmse <= 20:
		group = 3
	elif mmse >= 21 and mmse <= 25:
		group = 2
	elif mmse >= 26 and mmse <= 29:
		group = 1
	elif mmse >= 30:
		group = 0
	return group
	

def isDifferent2(group1, group2, counts):
	if group1 == None or group2 == None:
		return None
	if group1 == 0 and group2 == 0:
		counts[(0,0)] += 1
		return 0
	elif group1 == 1 and group2 == 1:
		counts[(1,1)] += 1
		return 0
	elif group1 == 0 and group2 == 1:
		counts[(0,1)] += 1
		return 0
	elif group1 == 0 and group2 == 2:
		counts[(0,2)] += 1
		return 1
	elif group1 == 0 and group2 == 3:
		counts[(0,3)] += 1
		return 1
	elif group1 == 0 and group2 == 4:
		counts[(0,4)] += 1
		return 1
	elif group1 == 1 and group2 == 2:
		counts[(1,2)] += 1
		return 1
	elif group1 == 1 and group2 == 3:
		counts[(1,3)] += 1
		return 1
	elif group1 == 1 and group2 == 4:
		counts[(1,4)] += 1
		return 1
	elif group1 == 2 and group2 == 3:
		counts[(2,3)] += 1
		return 1
	elif group1 == 3 and group2 == 4:
		counts[(3,4)] += 1
		return 1
	elif group1 == 2 and group2 == 4:
		counts[(2,4)] += 1
		return 1
	else:
		return None
		



class MultiDataset(Dataset):
	def __init__(self, root_dir, train, augment = False, dataset=None, list_ids=None, list_labels=None, list_infos=None, list_nbpersubj=None, list_uniqueids = None, val_size=None, fold=0, test = False, missing_data = False, inference=False):
		self.augment = augment
		self.missing = missing_data
		self.test = test
		self.inference = inference
		valstart = fold * val_size
		valstop = fold * val_size + val_size
		dict_visit_rev = {0:"bl",1:"m06",2:"m12",3:"m24"}
		dict_modality_rev = {0:"clin",1:"mr"}
		if (train == True) and (dataset == None):
			print('Building new dataset\tTRAINING DATA')
			self.root_dir = root_dir
			self.list = os.listdir(self.root_dir)
			self.list.sort(key=extractInfos)
			#print(len(self.list))
			
			last_mmse = dict()
			last_viscode = dict()
			first_mmse = dict()
			f = open("ADNIMERGE_.csv","r")
			for i,line in enumerate(f.readlines()):
				if i > 0 :
					fields = line.split(',')
					id_ = fields[0]
					viscode = int(fields[1])
					try:
						mmse = float(fields[-2])
					except ValueError:
						mmse = 100.
					if viscode == 0:
						first_mmse[id_] = mmse
					if (id_ not in last_mmse.keys()) and (viscode <= 72):
						last_mmse[id_] = mmse
						last_viscode[id_] = viscode
					elif (id_ in last_mmse.keys()) and (viscode > last_viscode[id_]) and (viscode <= 72) and (mmse != 100.):
						last_mmse[id_] = mmse
						last_viscode[id_] = viscode
					else:
						pass
			
			self.dico_subjs = {}
			self.list_files = []
			self.list_uniqueids = []
			self.list_allids = []
			self.list_alllabels = []
			self.list_allinfos = []
			self.list_nbpersubj = []
			
			nbpersubj = {}
			combcount = {(0,1):0,(1,2):0,(2,3):0,(1,3):0,(0,2):0,(0,3):0}
			combtotal = {(0,1):0,(1,2):0,(2,3):0,(1,3):0,(0,2):0,(0,3):0}
			counts={(0,0):0,(1,1):0,(0,1):0,(0,2):0,(0,3):0,(0,4):0,(0,5):0,(1,2):0,(1,3):0,(1,4):0,(1,5):0,(2,3):0,(2,4):0,(3,4):0}

			prev_id = extractInfos2(self.list[0])[3] #id of first subject in list
			subj_id = extractInfos2(self.list[0])[3] #id of first subject in list
			nbpersubj[subj_id] = 0
			i = 0
			k = 0	
			mmse_list = [None, None, None, None]
			diag_list = [None, None, None, None]
			combinations = {(0,1):None,(1,2):None,(2,3):None,(1,3):None,(0,2):None,(0,3):None}
			
			while i < len(self.list)-1:
				#print(self.list[i])
				#print(self.list[i+1])
				#print(i)
				subj_id = extractInfos2(self.list[i])[3] #id of first subject in list
				if subj_id == prev_id:
					infos = extractInfos2(self.list[i])
					diag_id = infos[0]
					visit_id = infos[1]
					modality_id = infos[2]
					class_id = infos[4]
					clin_file = self.list[i]
					mr_file = self.list[i+1]
					mmse = np.asarray(torch.load(self.root_dir+"/"+clin_file))[-2]
					if mmse == "":
						mmse = 0
					mmse = float(mmse)
					mmse_list[visit_id] = mmse
					diag_list[visit_id] = diag_id
					
					prev_id = infos[3]
					i += 2
				else:
					#print(mmse_list)
					nbpersubj[subj_id] = 0
					k += 1 #new subject id
					self.list_uniqueids.append(prev_id)
					self.dico_subjs[prev_id] = []
					for key in combinations.keys():
						g1 = mmse_group1(mmse_list[key[0]])
						g2 = mmse_group1(mmse_list[key[1]])						
						res = isDifferent2(g1, g2, counts)
						combinations[key] = res
						if res != None:
							clinf0 = diag_list[key[0]]+"_clin"+dict_visit_rev[key[0]]+"_"+prev_id+"_"+class_id+".pt"
							clin0 = np.asarray(torch.load(self.root_dir+"/"+clinf0))[:-1]
							clin0 = np.asarray([val if val != "" else "0" for val in clin0]).astype(np.float32)
							mrf0 = diag_list[key[0]]+"_mr"+dict_visit_rev[key[0]]+"_"+prev_id+"_"+class_id+".pt"
							mr0 = torch.load(self.root_dir+"/"+mrf0)
							#mr0 = mr0.astype(np.float32) / 255.
							clinf1 = diag_list[key[1]]+"_clin"+dict_visit_rev[key[1]]+"_"+prev_id+"_"+class_id+".pt"
							clin1 = np.asarray(torch.load(self.root_dir+"/"+clinf1))[:-1]
							clin1 = np.asarray([val if val != "" else "0" for val in clin1]).astype(np.float32)
							mrf1 = diag_list[key[1]]+"_mr"+dict_visit_rev[key[1]]+"_"+prev_id+"_"+class_id+".pt"
							mr1 = torch.load(self.root_dir+"/"+mrf1)
							#mr1 = mr0.astype(np.float32) / 255.
							"""
							plt.figure()
							plt.imshow(mr0[:,:,30],cmap="gray")
							plt.figure()
							plt.imshow(mr1[:,:,30],cmap="gray")
							plt.show()
							"""
							
							
							self.dico_subjs[prev_id].append({"files":(mr0, mr1, clin0, clin1, res),"label":res,"infos":(g1, g2, mmse_list[key[0]], mmse_list[key[1]], key[0], key[1], prev_id, class_id, diag_list[key[1]],last_mmse[prev_id],first_mmse[prev_id]),"id":prev_id})
							nbpersubj[prev_id] += 1
							combtotal[key[0],key[1]] += 1
							if res == 1:
								combcount[key[0],key[1]] += 1
					#print(combinations)
					mmse_list = [None, None, None, None]
					combinations = {(0,1):None,(1,2):None,(2,3):None,(1,3):None,(0,2):None,(0,3):None}
					#print("reset")
					prev_id = extractInfos2(self.list[i])[3] #id of first subject in list
				"""
				print(self.list_ids)
				print(self.list_labels)
				print("Change ",count1)
				print("No change ",count0)
				p = input("....")		
				"""

			count_viscode = {0:0,3:0,6:0,12:0,18:0,24:0,30:0,36:0,42:0,48:0,54:0,60:0,72:0}
			for key in last_viscode.keys():
				if key in self.list_uniqueids:
					count_viscode[last_viscode[key]] += 1
			print(count_viscode)
			#print("Total number of subjects: ",len(nbpersubj))
			#seed = np.random.randint(0,1024)
			seed = 42
			#print("SEED ", seed)
			random.seed(seed)
			random.shuffle(self.list_uniqueids)
			#print("WARNING:NO SHUFFLE BEFORE TRAIN/VAL/TEST SPLIT")
			for id_ in self.list_uniqueids :
				self.list_nbpersubj.append(nbpersubj[id_])
				for it in range(len(self.dico_subjs[id_])):
					self.list_files.append(self.dico_subjs[id_][it]["files"])
					self.list_alllabels.append(self.dico_subjs[id_][it]["label"])
					self.list_allinfos.append(self.dico_subjs[id_][it]["infos"])
					self.list_allids.append(self.dico_subjs[id_][it]["id"])
			
			teststop = round(0.2*len(self.list_uniqueids))
			maxidtest = 0
			for i in range(len(self.list_uniqueids[0:teststop])):
				maxidtest += self.list_nbpersubj[i]
			start = 0
			for i in range(len(self.list_uniqueids[0:valstart])):
				start += self.list_nbpersubj[i]
			stop = 0
			for i in range(len(self.list_uniqueids[0:valstop])):
				stop += self.list_nbpersubj[i]
				

			self.dataset = self.list_files[maxidtest:]
			self.list_ids = self.list_allids[maxidtest:]
			self.list_uniqueids = self.list_uniqueids[teststop:]
			self.list_labels = self.list_alllabels[maxidtest:]
			self.list_infos = self.list_allinfos[maxidtest:]
			self.test = self.list_files[:maxidtest]
			self.test_ids = self.list_allids[:maxidtest]
			self.test_labels = self.list_alllabels[:maxidtest]
			self.test_infos = self.list_allinfos[:maxidtest]
			self.test_uniqueids = self.list_allinfos[:teststop]
			#print(self.list_ids)
			if self.list_files[0:start] == None:
				self.data = self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
				self.infos = self.list_infos[stop:len(self.list_infos)]
				self.uniqueids = self.list_uniqueids[valstop:len(self.list_uniqueids)]
			
			elif self.list_files[stop:len(self.list_files)] == None:
				self.data = self.dataset[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
				self.infos = self.list_infos[0:start]
				self.uniqueids = self.list_uniqueids[0:valstart]
			
			else:
				self.data = self.dataset[0:start] + self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
				self.infos = self.list_infos[0:start] + self.list_infos[stop:len(self.list_infos)]
				self.uniqueids = self.list_uniqueids[0:valstart] + self.list_infos[valstop:len(self.list_uniqueids)]
			
				
		elif (train == True) and (dataset != None):
			print('Using pre-shuffled dataset\tTRAINING DATA')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_uniqueids = list_uniqueids
			self.list_labels = list_labels
			self.list_infos = list_infos
			self.list_nbpersubj = list_nbpersubj
			start = 0
			for i in range(len(self.list_uniqueids[0:valstart])):
				start += self.list_nbpersubj[i]
			stop = 0
			for i in range(len(self.list_uniqueids[0:valstop])):
				stop += self.list_nbpersubj[i]
			if self.dataset[0:start] == None:
				self.data = self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
				self.infos = self.list_infos[stop:len(self.list_infos)]
				self.uniqueids = self.list_uniqueids[valstop:len(self.list_uniqueids)]
			
			elif self.dataset[stop:len(self.dataset)] == None:
				self.data = self.dataset[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
				self.infos = self.list_infos[0:start]
				self.uniqueids = self.list_uniqueids[0:valstart]
			
			else:
				self.data = self.dataset[0:start] + self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
				self.infos = self.list_infos[0:start] + self.list_infos[stop:len(self.list_infos)]
				self.uniqueids = self.list_uniqueids[0:valstart] + self.list_infos[valstop:len(self.list_uniqueids)]

		elif (train == False) and (test == False):
			print('Using pre-shuffled dataset\tVALIDATION DATA')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_uniqueids = list_uniqueids
			self.list_labels = list_labels
			self.list_infos = list_infos
			self.list_nbpersubj = list_nbpersubj
			start = 0
			for i in range(len(self.list_uniqueids[0:valstart])):
				start += self.list_nbpersubj[i]
			stop = 0
			for i in range(len(self.list_uniqueids[0:valstop])):
				stop += self.list_nbpersubj[i]
			self.data = self.dataset[start:stop]
			self.labels = self.list_labels[start:stop]
			self.ids = self.list_ids[start:stop]
			self.infos = self.list_infos[start:stop]
			self.uniqueids = self.list_uniqueids[valstart:valstop]
			
		elif (train == False) and (test == True):
			print('Using test dataset\tTEST DATA')
			self.data = dataset
			self.ids = list_ids
			self.labels = list_labels
			self.infos = list_infos
			self.uniqueids = list_uniqueids

		print("TOTAL PAIRS ",len(self.ids))
		print("TOTAL UNIQUE SUBJS ",len(self.uniqueids))
		print("CHANGE ",self.labels.count(1))
		print("NO CHANGE ",self.labels.count(0))
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.data)
		
	def __getitem__(self, idx):
		'Generate one batch of data'
		infos = self.infos[idx]
		imgs, imgs2, clinic, clinic2, labels = self.data[idx]
		imgs = imgs.astype(np.float32) / 255.
		imgs2 = imgs2.astype(np.float32) / 255.
		imgs = exposure.rescale_intensity(imgs, in_range="image",out_range=(0.,1.))
		imgs2 = exposure.rescale_intensity(imgs2, in_range="image" ,out_range=(0.,1.))
		clin = clinic[3:-1] #1st 3 are demographics, last is MMSE value
		clin2 = clinic2[3:-1]
		dem = clinic[:3]

		if self.augment == True:
			
			sigma = torch.randint(low=0, high=8, size=(1,)).item()*0.1
			imgs = scipy.ndimage.gaussian_filter(imgs, sigma=sigma, mode='nearest')
			imgs2 = scipy.ndimage.gaussian_filter(imgs2, sigma=sigma, mode='nearest')
			
			angle = torch.randint(low=0, high=11, size=(1,)).item()
			angle2 = torch.randint(low=0, high=11, size=(1,)).item()
			neg = torch.randint(0,2,(1,)).item()
			neg2 = torch.randint(0,2,(1,)).item()
			if neg == 1:
				angle = - angle
			if neg2 == 1:
				angle2 = - angle2
			imgs= scipy.ndimage.interpolation.rotate(imgs, angle, axes=(1,2), reshape=False, mode='nearest')
			imgs2 = scipy.ndimage.interpolation.rotate(imgs2, angle2, axes=(1,2), reshape=False, mode='nearest')
			
			flip = torch.randint(0,5,(1,)).item()			
			if flip == 1:
				imgs = np.flip(imgs, 2).copy()	
				imgs2 = np.flip(imgs2, 2).copy()			
			
			ib = torch.randint(0,3,(1,)).item()
			ih = torch.randint(98,101,(1,)).item()
			pb, ph = np.percentile(imgs, (ib, ih))
			pb2, ph2 = np.percentile(imgs2, (ib, ih))

			imgs = exposure.rescale_intensity(imgs, in_range=(pb, ph) ,out_range=(0.,1.))
			imgs2 = exposure.rescale_intensity(imgs2, in_range=(pb2, ph2) ,out_range=(0.,1.))

		imgs = torch.as_tensor(imgs,dtype=torch.float32).view(1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
		imgs2 = torch.as_tensor(imgs2,dtype=torch.float32).view(1, imgs2.shape[0], imgs2.shape[1], imgs2.shape[2])
		clin = torch.as_tensor(clin,dtype=torch.float32)
		clin2 = torch.as_tensor(clin2,dtype=torch.float32)
		return imgs, imgs2, clin, clin2, dem, labels, infos
		
	def getShuffledDataset(self):
		return self.dataset, self.list_ids, self.list_labels, self.list_infos, self.list_nbpersubj, self.list_uniqueids
		
	def getTestDataset(self):
		return self.test, self.test_ids, self.test_labels, self.test_infos, self.test_uniqueids		
		
	def getSampler(self):
		class_sample_count = np.array(
		[len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
		total = len(self.labels)
		imin = np.argmin(class_sample_count)
		weight = 1. / class_sample_count
		#weight = 1 - (class_sample_count / total)
		#weight = np.array([0.1,1.0]) #we want ALL Change subjects (undersampling)
		samples_weight = np.array([3*weight[t] if t==imin else weight[t] for t in self.labels])
		#print(class_sample_count)
		#print(weight)
		#print(self.labels)
		#print(samples_weight)
		samples_weight = torch.from_numpy(samples_weight)
		samples_weigth = samples_weight.double()
		self.sampler = torch.utils.data.WeightedRandomSampler(samples_weight, 2*int(np.amin(class_sample_count)), replacement=False) #random undersampling
		return self.sampler

		
class TDSNet(nn.Module):
	def __init__(self):
		super(TDSNet, self).__init__()
		
		self.BN1_mri = torch.nn.BatchNorm3d(1)
		self.C1 = torch.nn.Conv3d(1, 8, 3, padding=1)
		self.BN2_mri = torch.nn.BatchNorm3d(8)
		self.LR1_mri = torch.nn.LeakyReLU()
		self.p_C1 = torch.nn.Conv3d(8, 8, 3, padding=0,stride=2)
		self.C2 = torch.nn.Conv3d(8, 16, 3, padding=1)
		self.BN3_mri = torch.nn.BatchNorm3d(16)
		self.LR2_mri = torch.nn.LeakyReLU()
		self.p_C2 = torch.nn.Conv3d(16, 16, 3, padding=0,stride=2)
		self.C3 = torch.nn.Conv3d(16, 32, 3, padding=1)
		self.BN4_mri = torch.nn.BatchNorm3d(32)
		self.LR3_mri = torch.nn.LeakyReLU()
		self.p_C3 = torch.nn.Conv3d(32, 32, 3, padding=0,stride=2)
		self.C4 = torch.nn.Conv3d(32, 32, 3, padding=1)
		self.BN5_mri = torch.nn.BatchNorm3d(32)
		self.LR4_mri = torch.nn.LeakyReLU()
		
		self.BN6_mri = torch.nn.BatchNorm3d(16)
		self.LR5_mri = torch.nn.LeakyReLU()
		self.Flat = torch.nn.Flatten()
		self.D1_mri = torch.nn.Linear(32*15*15*9, 512)
		self.BN7_mri = torch.nn.BatchNorm1d(512)
		self.LR6_mri = torch.nn.LeakyReLU()
		self.D2_mri = torch.nn.Linear(512, 512)
		self.BN8_mri = torch.nn.BatchNorm1d(512)
		self.LR7_mri = torch.nn.LeakyReLU()
		
		self.Pool = torch.nn.AvgPool3d(3,2)
		self.Drop1 = torch.nn.Dropout(0.7)
		self.Drop2 = torch.nn.Dropout(0.7)
		self.Out = torch.nn.Linear(512,1)
		self.Sig = torch.nn.Sigmoid()
		
	def forward_once(self, mri):
		#print("0",mri.shape)
		x = self.BN1_mri(mri)
		x = self.C1(x)
		x = self.BN2_mri(x)
		x = self.LR1_mri(x)
		x = self.p_C1(x)
		#print("1",x.shape)

		x = self.C2(x)
		x = self.BN3_mri(x)
		x = self.LR2_mri(x)
		x = self.p_C2(x)
		#print("2",x.shape)

		x = self.C3(x)
		x = self.BN4_mri(x)
		x = self.LR3_mri(x)
		x = self.p_C3(x)
		#print("3",x.shape)
		
		x = self.C4(x)
		x = self.BN5_mri(x)
		x = self.LR4_mri(x)
		#print("4",x.shape)
		
		return x
		
	def forward(self, left_mri, right_mri):
		l_mri = self.forward_once(left_mri)
		r_mri = self.forward_once(right_mri)
		diff_mri = torch.abs(torch.add(l_mri,torch.neg(r_mri)))

		x = self.Flat(diff_mri)
		x = self.D1_mri(x)
		x = self.BN7_mri(x)
		x = self.LR6_mri(x)
		x = self.Drop1(x)
		x = self.D2_mri(x)
		out = self.Drop2(x)
		out = self.Out(out)
		out = self.Sig(out)
		return x, out

class ClinNet(nn.Module):
	def __init__(self):
		super(ClinNet, self).__init__()
		
		self.BN1_cl = torch.nn.BatchNorm1d(8)
		self.D1_cl = torch.nn.Linear(8, 512)
		self.BN2_cl = torch.nn.BatchNorm1d(512)
		self.LR1_cl = torch.nn.LeakyReLU()
		self.D2_cl = torch.nn.Linear(512, 512)
		self.BN3_cl = torch.nn.BatchNorm1d(512)
		self.LR2_cl = torch.nn.LeakyReLU()
		
		self.BN4_cl = torch.nn.BatchNorm1d(512)
		self.LR3_cl = torch.nn.LeakyReLU()
		self.D3_cl = torch.nn.Linear(512, 8)
		self.BN5_cl = torch.nn.BatchNorm1d(8)
		self.LR4_cl = torch.nn.LeakyReLU() 
		
		self.BN6_cl = torch.nn.BatchNorm1d(3)
		self.D4_cl = torch.nn.Linear(3, 512)
		self.BN7_cl = torch.nn.BatchNorm1d(512)
		self.LR5_cl = torch.nn.LeakyReLU()
		self.D5_cl = torch.nn.Linear(512, 512)
		self.BN8_cl = torch.nn.BatchNorm1d(512)
		self.LR6_cl = torch.nn.LeakyReLU()
		self.D6_cl = torch.nn.Linear(512, 3)
		self.BN9_cl = torch.nn.BatchNorm1d(3)
		self.LR7_cl = torch.nn.LeakyReLU()
		
		self.Drop = torch.nn.Dropout(0.5)
		self.Out = torch.nn.Linear(11, 1)
		self.Sig = torch.nn.Sigmoid()
		
	def forward_once(self, cl):
		x = self.BN1_cl(cl)
		x = self.D1_cl(x)
		x = self.BN2_cl(x)
		x = self.LR1_cl(x)
		x = self.Drop(x)
		x = self.D2_cl(x)
		x = self.BN3_cl(x)
		x = self.LR2_cl(x)
		
		return x
		
	def forward(self, left_cl, right_cl, dem):
		l_cl = self.forward_once(left_cl)
		r_cl = self.forward_once(right_cl)
		diff_cl = torch.abs(l_cl-r_cl)
		
		x = self.BN4_cl(diff_cl)
		x = self.LR3_cl(x)
		x = self.D3_cl(x)
		x = self.BN5_cl(x)
		x = self.LR4_cl(x)
		
		d = self.BN6_cl(dem)
		d = self.D4_cl(d)
		d = self.BN7_cl(d)
		d = self.LR5_cl(d)
		d = self.D5_cl(d)
		d = self.BN8_cl(d)
		d = self.LR6_cl(d)
		d = self.D6_cl(d)
		d = self.BN9_cl(d)
		d = self.LR7_cl(d)
		
		cl = torch.cat((x,d),1)
		out = self.Drop(cl)
		out = self.Out(out)
		out = self.Sig(out)
		return cl, out
		
class MultiNet(nn.Module):
	def __init__(self):
		super(MultiNet, self).__init__()
		self.clin_module = ClinNet()
		self.mri_module = TDSNet()
		self.BN1_both = torch.nn.BatchNorm1d(523)
		self.LR1_both = torch.nn.LeakyReLU()
		self.D1_both = torch.nn.Linear(523, 512)
		self.BN2_both = torch.nn.BatchNorm1d(512)
		self.LR2_both = torch.nn.LeakyReLU()
		self.D2_both = torch.nn.Linear(512, 512)

		self.weight = torch.nn.Parameter(torch.FloatTensor(2, 512))
		nn.init.xavier_uniform_(self.weight)
		self.Soft = torch.nn.Softmax()
		
	def forward(self, l_mri, r_mri, l_cl, r_cl, dem):
		cl, _ = self.clin_module(l_cl, r_cl, dem)
		mri, _ = self.mri_module(l_mri, r_mri)
		x = torch.cat((mri, cl), 1)	
		x = self.BN1_both(x)
		x = self.LR1_both(x)
		x = self.D1_both(x)
		x = self.BN2_both(x)
		x = self.LR2_both(x)
		x = self.D2_both(x)
		#x = self.BN3_both(x)
		#x = self.LR3_both(x)
		#x = self.Drop(x)
		x = F.linear(F.normalize(x), F.normalize(self.weight))
		out = self.Soft(x)
		return x, out


batch_size = 40
num_classes = 2
epochs = 15
val_size = round(0.2*381) 

path = "./ADNI_full/data"
datatrain = MultiDataset(path, train=True, augment=False, val_size=val_size)
shuffled_dataset, ids, labels, infos, nbpersubj, uniqueids = datatrain.getShuffledDataset()

test_dataset, test_ids, test_labels, test_infos, test_uniqueids = datatrain.getTestDataset()
print("Example of test data: ")
print(test_ids[:10])

dataval = MultiDataset(path, train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels,list_infos = infos, list_nbpersubj = nbpersubj, list_uniqueids = uniqueids, val_size=val_size, missing_data=False)

nb_folds = (len(datatrain) + len(dataval)) // len(dataval)
print((len(datatrain) + len(dataval)) % val_size)
print("\nRunning training with "+str(nb_folds)+"-fold validation")


#p = input("\nPress Enter to continue\n")

for fold in range(nb_folds):

	datatrain = MultiDataset(path, train=True, augment=True, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, list_infos = infos, list_nbpersubj = nbpersubj, list_uniqueids = uniqueids, val_size=val_size, missing_data=False, fold=fold)
	sampler = datatrain.getSampler()	
	train_dataloader = DataLoader(datatrain, shuffle=True, num_workers=10,batch_size=batch_size, drop_last=True) #sampler mutually exclusive with shuffle

	dataval = MultiDataset(path, train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels,list_infos = infos, list_nbpersubj = nbpersubj, list_uniqueids = uniqueids, val_size=val_size, missing_data=False,fold=fold)
	sampler = dataval.getSampler()
	val_dataloader = DataLoader(dataval, shuffle=True, num_workers=10,batch_size=batch_size, drop_last=True)
	
	datatest = MultiDataset(path, train = False, augment=False, dataset = test_dataset, list_ids = test_ids, list_labels = test_labels, list_infos = test_infos, list_uniqueids = test_uniqueids, val_size=val_size, test= True, missing_data=False,fold=fold, inference =True)
	test_dataloader = DataLoader(datatest, shuffle=True, num_workers=0,batch_size=1, drop_last=True) # /!\ missing data


	
	# Create model
	tdsnet = MultiNet()
	tdsnet = torch.nn.DataParallel(tdsnet,device_ids=[0,1])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tdsnet = tdsnet.to(device)
	
	head = ArcMarginProduct()
	head = head.to(device)

	ce = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(tdsnet.parameters(), lr=0.005, weight_decay = 0.2)

	loss_history = []
	valloss_history = []
	acc_history = []
	valacc_history = []
	f1_history = []
	valf1_history = [] 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))
		start_time = time.time()  
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0

		tdsnet.train()
		for i, data in enumerate(train_dataloader):
			imgs,imgs2,clin, clin2, dem, label, _ = data
			optimizer.zero_grad()
			label = label.to(device)
			imgs = imgs.to(device)
			imgs2 = imgs2.to(device)
			clin = clin.to(device)
			clin2 = clin2.to(device)
			dem = dem.to(device)
			embeddings, out = tdsnet(imgs, imgs2, clin, clin2, dem)

			ce_loss = ce(embeddings, label)
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.argmax(out.cpu().detach().numpy(),axis=1)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)

		print("Training loss ",tot_train_loss/float(i+1))
		loss_history.append(tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		acc_history.append(float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		f1_history.append(float(tot_train_f1)/float(i+1))
		
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
		tdsnet.train(False)
		with torch.no_grad():
			for j, data in enumerate(val_dataloader):
				imgs,imgs2,clin, clin2, dem, label, _ = data
				label = label.to(device)
				imgs = imgs.to(device)
				imgs2 = imgs2.to(device)
				clin = clin.to(device)
				clin2 = clin2.to(device)
				dem = dem.to(device)
				embeddings,out = tdsnet(imgs, imgs2, clin, clin2, dem)				

				ce_loss = ce(embeddings, label)
				val_loss =  ce_loss #+ mse_loss
		
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.argmax(out.cpu().detach().numpy(),axis=1)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		valacc_history.append(float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
		valf1_history.append(float(tot_val_f1)/float(j+1))
				
		print("Time (s): ",(time.time() - start_time))
		
		if (epoch > 0) and (valloss_history[-1] < min(valloss_history[:-1])):
			torch.save(tdsnet.state_dict(), "Pos-Neg_Multi_loss_"+str(fold)+".pt")
			print("Lowest loss so far")
			print("Model saved")
		print("\n")
		tdsnet.train()

	d = [loss_history, valloss_history,acc_history,valacc_history, f1_history,valf1_history]
	export_data = zip_longest(*d, fillvalue = '')
	with open('clinical_adapt_'+str(fold)+'.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
		  wr = csv.writer(myfile)
		  wr.writerow(("loss", "valloss","acc","valacc", "f1","valf1"))
		  wr.writerows(export_data)
	myfile.close()
	
######### INFERENCE
	
	del tdsnet
	
	tdsnet = MultiNet()
	trained_model = "Pos-Neg_Multi_loss_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	tdsnet.load_state_dict(new_state_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tdsnet = tdsnet.to(device)

	total=0
	correct=0
	tot_val_loss=0
	tot_val_f1 = 0
	infosg1_ = []
	infosg2_ = []
	infosmmse1_ = []
	infosmmse2_ = []
	infoscombi1_ = []
	infoscombi2_ = []
	infosid_ = []
	infosclass_ = []
	infosdiag_ = []
	infosmmselast_ = []
	infosmmsefirst_ = []
	label_ = np.asarray([])
	predicted_ = np.asarray([])
	out_ = np.asarray([])
	tdsnet.train(False)

	with torch.no_grad():
		for j, data in enumerate(test_dataloader):
			imgs,imgs2,clin, clin2, dem, label, info = data
			label = label.to(device)
			imgs = imgs.to(device)
			imgs2 = imgs2.to(device)
			clin = clin.to(device)
			clin2 = clin2.to(device)
			dem = dem.to(device)
			_, out = tdsnet(imgs, imgs2, clin, clin2, dem)
		
			label = label.cpu().detach().numpy()
			label_ = np.concatenate((label_,label))
			predicted = np.argmax(out.cpu().detach().numpy(),axis=1)
			predicted_ = np.concatenate((predicted_, predicted))
			out = out[:,-1].cpu().detach().numpy()
			out_ = np.concatenate((out_, out))
			infosg1_ += info[0]
			infosg2_ += info[1]
			infosmmse1_ += info[2]
			infosmmse2_ += info[3]
			infoscombi1_ += info[4]
			infoscombi2_ += info[5]
			infosid_ += info[6]
			infosclass_ += info[7]
			infosdiag_ += info[8]
			infosmmselast_ += info[9]
			infosmmsefirst_ += info[10]
			

	total = len(label_)
	correct = np.sum(np.equal(predicted_, label_))
	tot_val_f1 = f1_score(label_, predicted_)
	confmat = sklearn.metrics.confusion_matrix(label_, predicted_)

	print("Test acc ",float(correct)/float(total))
	print("Test f1 ",float(tot_val_f1))
	print(confmat)
	tn, fp, fn, tp = confmat.ravel()
	
	
	f = open("results_multi_pos-neg"+str(fold)+".csv","w")
	f.write("G1G2\tMMSE1\tMMSE2\tCombination\tInterval\tId\tClass\tDiag\tLastMMSE\tFirstMMSE\tTrue\tPred\n")
	for i in range(len(label_)):
		seq = str(infosg1_[i].item())+str(infosg2_[i].item())+"\t"+str(infosmmse1_[i].item())+"\t"+str(infosmmse2_[i].item())+"\t"+str(infoscombi1_[i].item())+str(infoscombi2_[i].item())+"\t"+str(infoscombi2_[i].item() - infoscombi1_[i].item())+"\t"+str(infosid_[i])+"\t"+str(infosclass_[i])+"\t"+str(infosdiag_[i])+"\t"+str(infosmmselast_[i].item())+"\t"+str(infosmmsefirst_[i].item())+"\t"+str(label_[i])+"\t"+str(predicted_[i])+"\n"
		f.write(seq)
	f.close()
	
	
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_, out_, drop_intermediate=False)
	auc = sklearn.metrics.auc(fpr, tpr)
	print("AUC ",auc)
	
	f = open("roc_multi_pos-neg"+str(fold)+".csv","w")
	f.write("fpr\ttpr\tthresh\n")
	for i in range(len(fpr)):
		seq = str(fpr[i])+"\t"+str(tpr[i])+"\t"+str(thresholds[i])+"\n"
		f.write(seq)
	f.close()
	
	
	"""
	epochs_history = [l for l in range(len(dist0_history))]
	
	plt.figure()
	plt.plot(loss_history)
	plt.plot(valloss_history)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train loss', 'Val loss'], loc='upper left')

	plt.figure()
	plt.plot(f1_history)
	plt.plot(valf1_history)
	plt.ylabel('F1')
	plt.xlabel('Epoch')
	plt.legend(['Train F1', 'Val F1'], loc='upper left')
	plt.show()
	"""

		

	

		
