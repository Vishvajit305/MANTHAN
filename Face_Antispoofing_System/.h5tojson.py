import json
import sys
from collections import defaultdict
import re
import io
import numpy
from tables import *
	



h5file = open_file(r"path for .h5 file")

##########################################################################################




class converter:
	"""
	Converts the contents of an HDF5 file into JSON. 
	"""
	def __init__(self, input_file):
		self.file_name = re.sub(r'\.h5$','',r"path for .h5 file")
		self.groupParentDict = defaultdict(list)
		self.groupContentsDict = {}
		self.file = input_file
		self.allGroups = []
		for group in input_file.walk_groups():
			name = group._v_name
			parent = group._v_parent
			parent = parent._v_name
			self.allGroups.append(name)
			self.groupParentDict[parent].append(name)
			self.groupContentsDict[name] = {}

			for array in h5file.list_nodes(group, classname="Array"):
				array_name = array._v_name
				array_contents = array.read()
				array_info = {array_name : array_contents}
				self.groupContentsDict[name].update(array_info)

			for gp in h5file.list_nodes(group, classname="Group"):
				gp_name = gp._v_name
				gp_contents = {gp_name : self.groupContentsDict[gp_name]}
				self.groupContentsDict[name].update(gp_contents)

			for table in h5file.list_nodes(group, classname="Table"):
				table_name = table._v_name
				table_contents = table.read()
				table_info = {table_name : table_contents}
				self.groupContentsDict[name].update(table_info)	

	def jsonOutput(self):
		alpha = self.groupContentsDict

		json_file_name = self.file_name + '.json' 
		with io.open(json_file_name, 'w', encoding='utf-8') as f:
			record = json.dumps(alpha,cls=NumpyAwareJSONEncoder)
			f.write(re.UNICODE(json.dumps(alpha, cls=NumpyAwareJSONEncoder, ensure_ascii=False)))
		f.close()
		return 

	def Groups(self):
		return json.dumps(self.allGroups, cls=NumpyAwareJSONEncoder)

	def subgroups(self, group):
		return json.dumps(self.groupParentDict[group], cls=NumpyAwareJSONEncoder)

	def groupContents(self, group):
		info = self.groupContentsDict[group]
		return json.dumps(info, cls=NumpyAwareJSONEncoder)



########################################################################################################

class NumpyAwareJSONEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, numpy.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


#######################################################################################################

json_data = converter(h5file)
contents = json_data.jsonOutput() 
