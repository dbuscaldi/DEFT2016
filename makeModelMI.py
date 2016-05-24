#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import math

import pickle
import gzip

import numpy as np
from numpy import linalg

class Model:
	#a model contains the indices, widx and lidx, and the LTM matrix
	#we use the model to serialize it using cPickle
	def __init__(self, wordict, labeldict, M):
		self.wordict=wordict
		self.labeldict=labeldict
		self.M=M
	
	def setWordFrequencies(self, df, N):
		self.freqs=df
		self.N=N
	
	def getWordFrequencies(self):
		return self.freqs
		
	def getMatrix(self):
		return self.M
	
	def getWordDict(self):
		return self.wordict
		
	def getLabelDict(self):
		return self.labeldict
	
class Notice:
	id=""
	text=""
	taggedtext=""
	labels=""

	def __init__(self, id, labels):
		self.id=id
		self.labels=labels
	
	def __hash__(self):
		return hash((self.id))
	
	def __eq__(self, other):
		return (self.id) == (other.id)
	
	def setTags(self, str):
		self.taggedtext=str
		
	def setText(self, str):
		self.text=str
	
	def write(self):
		print self.id, self.labels, self.taggedtext, self.text
		
collection = {}

idx={} #index document id -> numerical id for matrices
widx={} #word -> numerical word id for matrices
lidx={} #label -> numerical label id for matrices
wdict={} #term-document matrix
ldict={} #label-document matrix
"""
print "loading Google freqs"
gfreq={}
gMf=0 #google maxfreq
gf=codecs.open("unigrams-fr.dat", "r")
for line in gf.xreadlines():
	els=line.strip().split('\t')
	gfreq[els[0]]=int(els[1])
	if int(els[1]) > gMf:
		gMf=int(els[1])
gf.close()
"""
print "parsing data"
#donner le repertoire base où se trouvent les dossiers pre/ et txt/ et le fichier ref
counter=0
root=sys.argv[1]
refFile=os.path.join(root, "ref")
ref=codecs.open(refFile, "r", "utf-8")
for line in ref.xreadlines():
	els=line.strip().split("\t")
	id=els[0]
	idx[id]=counter
	counter+=1
	labels=els[1].split(";")
	n = Notice(id, labels)
	collection[id]=n

N=counter+1 #we use this as the size of the collection

preDir=os.path.join(root, "pre")
for file in os.listdir(preDir):
    if file.endswith(".pre"):
        id=file.replace(".pre", "")
        ffname=os.path.join(preDir, file)
        tf=codecs.open(ffname, "r", "utf-8")
        lines=tf.readlines()
        tf.close()
         
        n=collection[id]
        n.setTags(' '.join(lines))

txtDir=os.path.join(root, "txt")
for file in os.listdir(txtDir):
    if file.endswith(".txt"):
        id=file.replace(".txt", "")
        ffname=os.path.join(txtDir, file)
        tf=codecs.open(ffname, "r", "utf-8")
        lines=tf.readlines()
        tf.close()
         
        n=collection[id]
        n.setText(' '.join(lines))

#make term-document matrix and label-document matrix to extract MI     
wordid=0
labelid=0
for n in collection.values():
	ttxt=n.taggedtext
	tokens=ttxt.split(' ')
	docid=n.id
	id=idx[docid]
	for t in tokens:
		items=t.split('/')
		if re.match("[na].+", items[1]): #nav si on veut aussi les verbes
			try:
				dlist=wdict[items[0]]
			except KeyError:
				dlist=set([]) #let's use set since we're interested if the word appears or not in the document
				widx[items[0]]=wordid #update word index
				wordid+=1
			dlist.add(id)
			wdict[items[0]]=dlist
			#print items[0], items[1]
	for l in n.labels:
		l=l.strip()
		try:
			dlist=ldict[l]
		except KeyError:
			lidx[l]=labelid #update label index
			labelid+=1
			dlist=[]
		dlist.append(id)
		ldict[l]=dlist

"""
#word similarities
wsim={}
wlist=[]
wlist=wdict.keys()
for i in xrange(len(wlist)):
	w=wlist[i]
	wdocs=wdict[w]
	wf=len(wdocs)
	if i < len(wlist)-1:
		for k in wlist[(i+1):]:
			kdocs=wdict[k]
			kf=len(kdocs)
			isect=set(wdocs) & set(kdocs)
			il=len(isect)
			if il > 0 and wf > 3 and kf > 3:
				mi=math.log1p(N*il)-math.log1p(wf*kf)
				try:
					mv=wsim[w]
				except KeyError:	
					mv=[]
				mv.append((k, mi))
				wsim[w]=mv	
"""

#init label-term matrix (rows=labels, columns=terms)
LTM=np.zeros((labelid, wordid))

print "calculating MI scores"
#calculate MIs
for l in ldict.keys():
	#l is the label
	lf=len(ldict[l])
	ldl=ldict[l]
	if len(ldl) < 5: continue # set a threshold on the frequency of labels
	for k in wdict.keys(): #word
		#if l== k : print l, k
		kdl=wdict[k]
		kf=len(kdl)
		isect=set(ldl) & set(kdl)
		il=len(isect)
		if il > 3: #set a threshold on the frequency of mutual occurrence
			mi=math.log1p(N*il)-math.log1p(lf*kf)
			#mi=math.log1p(N*il)-math.log1p(kf*kf) #using p(label|word)
			if mi > 0:
				i=lidx[l]
				j=widx[k]
				LTM[i][j]=mi
				#try:
				#	LTM[i][j]=mi*(math.log(gMf)-math.log(gfreq[k])) #smooth MI by IDF in Google (to reduce importance of very frequent words)
				#except KeyError:
				#	pass
				#optionally: extend mi to words that are similar to k but not co-occurring with l
				#(maybe computationally expensive...)
"""
# apply filter to LTM
print "normalizing weight matrix"

nLTM=np.zeros((labelid, wordid))
for l in ldict.keys():
	i=lidx[l]
	labelvec=LTM[i]
	for k in xrange(len(labelvec)):
		if i <> k:
			wvec=LTM[:, k]
			wiset=np.flatnonzero(wvec)
			wnzv=[]
			for j in wiset:
				wnzv.append(wvec[j])
			
			if sum(wnzv) <> 0:
				nzavg=np.mean(wnzv)
				nzstd=np.std(wnzv)
			
				if LTM[i,k] >= nzstd+nzavg:
					#use the weight if and only if the difference is statistically significative 
					#nLTM[i,k] = LTM[i,k] #binarized version: nLTM[i,k]=1.0
					nLTM[i,k] = 1.0
					#else: leave 0
				
m=Model(widx, lidx, nLTM) #use LTM for the standard
"""
#SVD and LSA
U, S, Vt = linalg.svd(LTM, full_matrices=False)

print S[:100]

#print linalg.norm(S) 95 for archeo, it seems 100 is a reasonable parameter

Sprime = np.zeros(len(S))

print S.shape, Sprime.shape

#reducing eigenvector dimensionality
i=0
for k in S[:100]:
	Sprime[i]=k
	i+=1

Sk= np.diag(Sprime)

print U.shape, Sk.shape, Vt.shape

rM= np.dot(U, np.dot(Sk, Vt))
#print np.allclose(LTM, rM)

#m=Model(widx, lidx, LTM)
m=Model(widx, lidx, rM) #use LSA matrix
m.setWordFrequencies(wdict, len(collection))

print "Saving model to model.pklz"
f = gzip.open('model.pklz','wb')
pickle.dump(m,f)
f.close()

# to create an inverse index:
#b = {v: k for k, v in a.iteritems()}
