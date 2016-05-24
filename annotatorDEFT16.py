#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import math

import pickle
import gzip
import argparse

import numpy as np
from scipy.spatial.distance import cosine

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.lang import *
from whoosh.analysis import *

from whoosh.qparser import QueryParser

from nltk.util import ngrams

class Model: #needed for the unpickle
	#a model contains the indices, widx and lidx, and the LTM matrix
	#we use the model to serialize it using cPickle
	def __init__(self, wordict, labeldict, M):
		self.wordict=wordict
		self.labeldict=labeldict
		self.M=M
	
	def setWordFrequencies(self, df):
		self.freqs=df
	
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
	assignedlabels=""
	
	def __init__(self, id):
		self.id=id
		self.assignedlabels=set([])
		
	#def __init__(self, id, labels):
	#	self.id=id
	#	self.labels=labels
	#	self.assignedlabels=set([])
	
	def __hash__(self):
		return hash((self.id))
	
	def __eq__(self, other):
		return (self.id) == (other.id)
	
	def setTags(self, strng):
		self.taggedtext=strng
		
	def setText(self, strng):
		self.text=strng
	
	def setAssignedLabels(self, labels):
		self.assignedlabels=self.assignedlabels | labels 
	
	def write(self):
		print self.id, self.labels, self.taggedtext, self.text

collection = {} #the collection to annotate

widx={} #word -> numerical word id for matrices
lidx={} #label -> numerical label id for matrices

parser = argparse.ArgumentParser(description='Annotate text using a MI-based model.')
parser.add_argument('model', metavar='model',
                   help='the .pklz file containing the model')
parser.add_argument('dir', metavar='dir',
                   help='the directory containing the files to annotate (pre/txt format)')
parser.add_argument('--repo', dest='repo', help='index to be used')
parser.add_argument('--mode', dest='mode', help='mode: h hybrid, mi mutual information only, ir yasemir only')

args = parser.parse_args()


print "reading files in directory", args.dir
counter=0
root=args.dir
"""
#no ref file for the test set
refFile=os.path.join(root, "ref")
ref=codecs.open(refFile, "r", "utf-8")
for line in ref.xreadlines():
	els=line.strip().split("\t")
	id=els[0]
	counter+=1
	labels=els[1].split(";")
	n = Notice(id, labels)
	collection[id]=n
"""

preDir=os.path.join(root, "pre")
for file in os.listdir(preDir):
	if file.endswith(".pre"):
		id=file.replace(".pre", "")
		ffname=os.path.join(preDir, file)
		tf=codecs.open(ffname, "r", "utf-8")
		lines=tf.readlines()
		tf.close()
		
		n = Notice(id)
		n.setTags(' '.join(lines))
		collection[id]=n

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

print "size of collection: " + str(len(collection))
print "annotating..."

if args.mode=="h" or args.mode=="mi":
	print "loading model "+args.model

	f = gzip.open(args.model,'rb')
	m=pickle.load(f)
	f.close()

	widx = m.getWordDict()
	lidx = m.getLabelDict()
	iwidx = {v: k for k, v in widx.iteritems()} #inverse word index
	ilidx = {v: k for k, v in lidx.iteritems()} #inverse label index

	LTM=m.getMatrix()
			
	#print LTM.shape, len(lidx), len(widx)

	wordfreqs=m.getWordFrequencies() #word frequencies over the training collection

	for id in collection.keys():
		found_labels=set([])
		vec=np.zeros(len(widx)) # the vector associated to this document
		notice=collection[id]
		ttxt=notice.taggedtext
		tokens=ttxt.split(' ')
		notice_indices=[] #not null vec indices
		for t in tokens:
			items=t.split('/')
			if re.match("[na].+", items[1]): #nav si on veut aussi les verbes
				try:
					index=widx[items[0]]
					vec[index]=1.0 #we use a boolean vector
					notice_indices.append(index) 
					#vec[index]=vec[index]+1 #we weight the vector using the tf
				except KeyError:
					continue #move to next token, this one is not in the dictionary
					
		ratings=[]
		for i in xrange(len(lidx)):
			labelvec=LTM[i]
			s=0
			for k in notice_indices: #this is more effective than normalizing the whole matrix
				if LTM[i,k] == 0: continue
			
				wvec=LTM[:, k]
				wiset=np.flatnonzero(wvec)
				wnzv=[]
				for j in wiset:
					wnzv.append(wvec[j])
				nzavg=np.mean(wnzv)
				nzstd=np.std(wnzv)
			
				#print ilidx[i], codecs.encode(iwidx[k], "utf-8")
				if LTM[i,k] > (0.5*nzstd)+nzavg:
					#print "->", LTM[i,k], nzavg
					s+=vec[k]*labelvec[k]
					#s+=1 #binarization
			
			#score=np.dot(vec, labelvec) #cosine, np.dot
			#score=(1-cosine(vec, labelvec)) #with cosine 0 means they are the same, 1 they are completely different
			#if score > 0 or s > 0: print "final score for:", ilidx[i], score, s
		
			if not math.isnan(s):
				ratings.append((ilidx[i], s))
	
		ratings.sort(key=lambda x : -x[1])
		for r in ratings[:20]:
			if r[1] > 0:
				found_labels.add(r[0])
		#print ratings[:5]
		notice.setAssignedLabels(found_labels)
		collection[id]=notice

if args.mode=="ir" or args.mode=="h":
	ix = open_dir(args.repo)
	with ix.searcher() as searcher:
		for k in collection.keys():
			found_labels=set([])
			notice=collection[k]
			sentence = notice.text
			#reflabels=set(notice.labels)
			n = 3
			trigrams = ngrams(sentence.split(), n)
			for grams in trigrams:
				#print grams
				query = QueryParser("label", schema=ix.schema).parse(' '.join(grams))
				results = searcher.search(query, limit=20)
				if len(results) > 0:
					ass_label=results[0]["label"]
					found_labels.add(codecs.encode(ass_label.lower(), "utf-8"))
					
			notice.setAssignedLabels(found_labels)
			collection[k]=notice


print "writing results..."
for k in collection.keys():
	notice=collection[k]
	alabels=notice.assignedlabels
	sys.stdout.write(k)
	sys.stdout.write('\t')
	try:
		sys.stdout.write(codecs.encode(';'.join(alabels), "utf-8"))
	except:
		sys.stdout.write(';'.join(alabels))
	sys.stdout.write('\n')
	
"""
print "evaluating..."
scores={}
sum_prec=0
sum_rec=0
for k in collection.keys():
	notice=collection[k]
	labels=set(notice.labels)
	alabels=notice.assignedlabels
	#print labels
	#print alabels
	ll=len(labels)
	all=len(alabels)
	isect= labels & alabels
	il = len(isect)
		
	if ll > 0:
		recall=float(il)/float(ll)
	else: recall=0.0
	if all > 0:
		precision=float(il)/float(all)
	else: precision= 0.0
		
	scores[k]=(recall, precision)
	sum_prec+=precision
	sum_rec+=recall
	
print "id, recall, precision:"
for k in scores.keys():
	print k, scores[k][0], scores[k][1]
	
prec= float(sum_prec)/float(N)
rec= float(sum_rec)/float(N)
print "average precision:" , prec
print "average recall:" , rec
print "F-measure:", 2* prec*rec/(prec+rec)
"""