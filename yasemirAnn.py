#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import rdflib
import argparse

from rdflib import OWL, RDFS

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.lang import *
from whoosh.analysis import *

from whoosh.qparser import QueryParser

from nltk.util import ngrams

class Notice:
	id=""
	text=""
	taggedtext=""
	labels=""
	assignedlabels=""

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
	
	def setAssignedLabels(self, labels):
		self.assignedlabels=labels
	
	def write(self):
		print self.id, self.labels, self.taggedtext, self.text
		
collection = {} #the collection to annotate

parser = argparse.ArgumentParser(description='Index ontology with labels and annotate text.')
parser.add_argument('ontology', metavar='ontology',
                   help='the RDF ontology containing the terminology')
parser.add_argument('--index', dest='idx', help='index to be created')
parser.add_argument('--ann', dest='annf', help='directory containing the files to annotate (pre, txt, ref format)')
parser.add_argument('--repo', dest='repo', help='index to be used')

args = parser.parse_args()

print args.ontology

g = rdflib.Graph()
result = g.parse(args.ontology)

ana = LanguageAnalyzer("fr")
    
if args.idx <> None:
	if not os.path.exists(args.idx): os.mkdir(args.idx)
	schema = Schema(concept=ID(stored=True), label=TEXT(stored=True, analyzer=ana))
	ix = create_in(args.idx, schema)
	writer = ix.writer()
	
	print "Schema created, indexing labels..."
	
	for subj, pred, obj in g:
		#print subj, pred, obj
		if str(pred).endswith("Label"):
			#print subj.n3(), obj.n3()
			utflabel=(obj.n3()).replace("@fr", "")
			utflabel=utflabel.strip('"')
			writer.add_document(concept=subj.n3(), label=utflabel)
	
	writer.commit()
	print "Indexing complete"
	sys.exit(0)
	
if args.annf <> None and args.repo <> None:
	print "reading files in directory", args.annf
	counter=0
	root=args.annf
	refFile=os.path.join(root, "ref")
	ref=codecs.open(refFile, "r", "utf-8")
	for line in ref.xreadlines():
		els=line.strip().split("\t")
		id=els[0]
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
			
	print "annotating..."
	
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
					"""
					if ass_label.lower() in reflabels:
						print "+"+' '.join(grams), ass_label, results[0].score
					else:
						print "-"+' '.join(grams), ass_label, results[0].score
					"""
					found_labels.add(ass_label.lower())
					
			notice.setAssignedLabels(found_labels)
			collection[k]=notice
	#evaluation:
	
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
	
else:
	print "Both annotation file and index repository must be specified"
