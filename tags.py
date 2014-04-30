from lxml import etree
from lxml import html
import cPickle as pickle
from collections import OrderedDict
import time
import pandas as pd
import numpy
import sklearn
import scipy
import matplotlib

class Torrent(object):
	id = 0
	title = ''
	magnet = ''
	size = 0
	seeders = 0
	leechers = 0
	upvotes = 0
	downvotes = 0
	uploaded = ''
	nfo = ''
	comments = []

	def printTorrent(self):
		print self.id, self.title.encode('utf-8'), self.upvotes, self.downvotes

all_torrents = {}
discarded_torrents = 0

# method for printing a tabbled XML tree
def printXML(root, depth):
	for n in range(0, depth):
		print " ",
	print depth,
	print root.tag, root.get("class"), root.text
	for child in root:
			printXML(child, depth+1)

def parseXML(root):
	global discarded_torrents
	torrents = root.findall("torrent");
	for t in torrents:
		next = Torrent()
		next.id = int(t.find("id").text)
		next.title = t.find("title").text
		next.magnet = t.find("magnet").text
		try:
			next.size = int(t.find("size").text)
			next.seeders = int(t.find("seeders").text)
			next.leechers = int(t.find("leechers").text)
			next.upvotes = int(t.find("quality")[0].text)
			next.downvotes = int(t.find("quality")[1].text)
			next.uploaded = t.find("uploaded").text
			next.nfo = t.find("nfo").text
#			for comment in t.find("comments").findall("comment"):
#				next.comments.append(comment.find("when").text)
				# comments also have a "what" field		except:
		except:	
			# must be the poor data file
			pass

		if next.upvotes != 0 or next.downvotes != 0:
			all_torrents[next.id] = next
		else:
			discarded_torrents += 1
		root.remove(t)

############## Loading torrents from xml file. Takes about 3 hours. Hopefully don't need it again.
#if __name__ == "__main__":
#	start = time.time()
#	print "Starting lxml parse"
##	xml = etree.parse("rich.short.xml")
##	xml = etree.parse("poor3.corrected.xml")
#	xml = etree.parse("rich.corrected.xml")
#	print "Tree created in", (time.time()-start), "seconds."
##	printXML(xml.getroot(), 0)
#
#	parseXML(xml.getroot())
#	print "XML parsed in", (time.time()-start), "seconds."
#
#	print len(all_torrents), "torrents kept"
#	print 100.0 - (100.0 * (float(discarded_torrents) / float(discarded_torrents + len(all_torrents)))), "percent of torrents kept"
#
#	pickle.dump(all_torrents, open('torrents.p', 'wb'))
#	print "Torrents saved in", (time.time()-start), "seconds."
#
#	del xml
#	del all_torrents
#	print "Objects deleted in", (time.time()-start), "seconds."

if __name__ == "__main__":
	start = time.time()
	all_torrents = pickle.load(open('torrents.p', 'rb'))
	print "Torrents reloaded in", (time.time()-start), "seconds."

	all_torrents = OrderedDict(sorted(all_torrents.iteritems(), key = lambda torrent: (torrent[1].upvotes + torrent[1].downvotes), reverse = False))
	for t_id in all_torrents:
		all_torrents[t_id].printTorrent()
	print len(all_torrents), "torrents found."
	print "Torrents printed in", (time.time()-start), "seconds."
