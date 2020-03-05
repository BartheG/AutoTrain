import sys,os

import xml.etree.ElementTree as ET

def findInDir(pathdir,ext):
	return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]

class ParseXml:
	def __init__(self, dirname):
		if not os.path.exists(dirname):
			print('Error: Input directory not found')
			raise FileNotFoundError
		self.dirname=dirname

	def read(self,):
		self.xmllist=findInDir(self.dirname,'.xml')
		if len(self.xmllist)<=0:
			print('No label modifications')
			return False
		return True

	def mformat(self,toformat):
		return 'item {\n  name: "'+toformat[0]+'"\n  id: '+toformat[1]+'\n}\n\n'

	def run(self,):
		if not self.read():
			return
		xmllabels=set([ name.text for xml in self.xmllist for name in ET.parse(xml).getroot().findall('./object/name') ])
		with open('labelmap.pbtxt', 'w') as outfile:
			for it,i in enumerate(xmllabels):
				outfile.write(self.mformat( ( str(i),str(it+1) ) ))
def main():
	if len(sys.argv)!=2:
		print('Error: Invalid number of arguments')
		raise ValueError
	ParseXml(sys.argv[1]).run()

if __name__ == "__main__":
	main()