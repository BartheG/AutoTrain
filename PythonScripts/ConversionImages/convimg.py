import sys
import os
import cv2
import numpy as np

def findInDir(pathdir,ext):
	return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]

class ConvImg:
	def __init__(self,dirname,n_size):
		self.n_size=n_size
		if not os.path.exists(dirname):
			print('Error: Input directory not found')
			raise FileNotFoundError
		self.dirname=dirname
		self.img_list=[]
		self.fileexttofind=[ '.jpg','.jpeg','.png' ]
		self.parseSize()
		self.mkdir()

	def parseSize(self,):
		if not '*' in self.n_size:
			print('Error: Wrong size format')
			raise ValueError
		self.sizetoresize=[int(s) for s in self.n_size.split('*') if s.isdigit()]
		if len(self.sizetoresize)!=2 or\
			self.sizetoresize[0]<=0 or self.sizetoresize[1]<=0 or\
			self.sizetoresize[0]>10000 or self.sizetoresize[1]>10000:
			print('Error: Wrong size format')
			raise ValueError

	def mkdir(self,):
		try:
			self.dirtosave='converted_'+str(self.n_size)+'_img'
			os.mkdir(self.dirtosave)
		except FileExistsError:
			pass

	def getImg(self,):
		for i in self.fileexttofind:
			self.img_list+=findInDir(self.dirname,i)

	def run(self,):
		self.getImg()
		for img in self.img_list:
			t_img=cv2.imread(img)
			if type(t_img)==np.ndarray:
				r_img=cv2.resize(
					t_img,
					dsize=(
						int(self.sizetoresize[0]),
						int(self.sizetoresize[1])
					)
				)
				cv2.imwrite(self.dirtosave+'/'+img.split('/')[-1],r_img)
		print(self.img_list)

def main():
	if len(sys.argv)!=3:
		print('Error: Incorrect number of arguments')
		raise ValueError
	ConvImg(sys.argv[1],sys.argv[2]).run()

if __name__ == "__main__":
	main()