import os,sys
import shutil

def findInDir(pathdir,ext):
	return [ (os.path.join(root, file),nameWithouthExt(os.path.join(root, file),ext)+str('.xml')) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) and os.path.exists(nameWithouthExt(os.path.join(root, file),ext)+str('.xml')) ]

def nameWithouthExt(path,ext):
	n_path=list(path)
	for _ in ext:
		n_path.pop(-1)
	return str(''.join(n_path))

class OrganizeData:
	def __init__(self,dirname):
		if not os.path.exists(dirname):
			print('Error: Input directory not found')
			raise FileNotFoundError
		self.dirname=dirname
		self.validext=[ '.jpg','.jpeg','.png','.JPG','.PNG' ]
		self.valid_img=[]
		self.mkdir()

	def mkdir(self):
		path=[ './test/','./train/','./test/data','./train/data','./test/labels','./train/labels' ]
		for name in path:
			try:
				os.mkdir(name)
			except FileExistsError:
				pass

	def runIntoDir(self,):
		for ext in self.validext:
			self.valid_img+=findInDir(self.dirname,ext)

	def run(self,):
		self.runIntoDir()
		self.nbtest=int(round(len(self.valid_img)*0.2))
		self.nbtrain=int(len(self.valid_img)-self.nbtest)
		print('Total files:',len(self.valid_img),'Nb tests:',self.nbtest,'Nb train:',self.nbtrain)
		for i in range(self.nbtest):
			shutil.copy( self.valid_img[i][0], './test/data/'+(self.valid_img[i][0].split('/')[-1]) )
			shutil.copy( self.valid_img[i][1], './test/labels/'+(self.valid_img[i][1].split('/')[-1]) )
		for i in range(self.nbtrain):
			shutil.copy( self.valid_img[i+(self.nbtest-1)][0], './train/data/'+(self.valid_img[i+(self.nbtest-1)][0].split('/')[-1]) )
			shutil.copy( self.valid_img[i+(self.nbtest-1)][1], './train/labels/'+(self.valid_img[i+(self.nbtest-1)][1].split('/')[-1]) )
		#print(self.valid_img)


def main():
	if len(sys.argv)!=2:
		print('Error: invalid number of arguments')
		raise ValueError
	OrganizeData(sys.argv[1]).run()

if __name__ == "__main__":
	main()