import os,sys

import cv2

def findInDir(pathdir,ext):
	return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]

class MovieCut:
	def __init__(self,name):
		self.isfile=False
		if not os.path.exists(name):
			print('Error: Input file/directory not found')
			raise FileNotFoundError
		if os.path.isfile(name):
			self.isfile=True
		else:
			self.fileext=[ '.mov','.mp4' ]
		self.mkdir()
		self.path=name
		self.video_list=[]

	def mkdir(self,):
		try:
			self.dirtosave='exported_frame'
			os.mkdir(self.dirtosave)
		except FileExistsError:
			pass

	def getNameOfFrame(self,path,num):
		videoname=path.split('/')[-1]
		videoext=videoname.split('.')
		n_videoname=''
		for i in range(len(videoext)-1):
			n_videoname+=videoext[i]
		return self.dirtosave+'/'+n_videoname+'_'+str(num)+'.jpg'

	def export(self,path):
		self.t_video=cv2.VideoCapture(path)
		frame_count=0
		rt_read=self.t_video.read()
		while rt_read[0]:
			cv2.imwrite(self.getNameOfFrame(path,frame_count),rt_read[1])
			rt_read=self.t_video.read()
			frame_count+=1

	def run(self,):
		if self.isfile:
			self.export(self.path)
			return
		for i in self.fileext:
			self.video_list+=findInDir(self.path,i)
		for name in self.video_list:
			self.export(name)

def main():
	if len(sys.argv)!=2:
		print('Error: Incorrect number of arguments')
		raise ValueError
	MovieCut(sys.argv[1]).run()

if __name__ == "__main__":
	main()