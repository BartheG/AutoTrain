import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys

def findInDir(pathdir,ext):
	return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]

class ReadAndConvert:
	def __init__(self,dirname):
		if not os.path.exists(dirname):
			print('Error: Invalid input directory')
			raise FileNotFoundError
		self.dirname=dirname
		self.t_df=[]

	def read(self,):
		self.xmllist=findInDir(self.dirname,'.xml')
		if len(self.xmllist)<=0:
			print('No csv modifications')
			return False
		return True

	def run(self,):
		if not self.read():
			return
		for xml in self.xmllist:
			try:
				root=ET.parse(xml).getroot()
				for obj in root.findall('object'):
					this=[
						root.find('filename').text,
						int(root.find('size')[0].text),
						int(root.find('size')[1].text),
						obj.find('name').text,
						int(float(obj.find('./bndbox/xmin').text)),
						int(float(obj.find('./bndbox/ymin').text)),
						int(float(obj.find('./bndbox/xmax').text)),
						int(float(obj.find('./bndbox/ymax').text))
					]
					self.t_df.append(this)
			except:
				pass

		df=pd.DataFrame(self.t_df,columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
		df.to_csv('labeldata.csv')

def main():
	if len(sys.argv)!=2:
		print('Erorr: Invalid number of arguments')
		raise ValueError
	ReadAndConvert(sys.argv[1]).run()

if __name__ == "__main__":
	main()