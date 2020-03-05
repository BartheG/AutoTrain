import json
import os,sys
import shutil

class JsonParser:
	def __init__(self,path):
		self.path = path
		if not os.path.exists(path):
			print('Error: JSON file not found')
			raise FileNotFoundError
		try: self.data = json.load( open(self.path) )
		except: raise TypeError
		self.out_name=self.data['name']+str(self.data['version'])
		self.mkdir( self.out_name )
		os.chdir(self.out_name)
		print('Output directory created')

	def mkdir(self,name):
		try:
			os.mkdir(name)
		except FileExistsError:
			pass

	def buildmodule(self,tobuild):
		sys.path.append('../'+tobuild['import_path'])
		m_module = __import__(tobuild['module_name'])
		m_class = getattr(m_module,tobuild['class_name'])
		return m_class,tobuild['args']

	def buildapp(self,tobuild):
		return\
			'../'+tobuild['app_path'],\
			' '.join([ '{} {}'.format(it,i) for it,i in tobuild['args'].items() ])

	def parse(self,i):
		if i['module'] and "command" in i.keys():
			return (True,self.buildmodule(i),i['command'])
		elif i['module']:
			return (True,self.buildmodule(i),None)
		elif "command" in i.keys():
			return (False,self.buildapp(i),i['command'])
		else:
			return (False,self.buildapp(i),None)

	def run(self,):
		n_data = { i:self.data[i] for i in self.data if 'step' in i }
		f_data = {}
		for it,i in n_data.items():
			f_data[it]=self.parse(i)
		return f_data

class Executor:
	def __init__(self):
		super().__init__()
		self.command={ "rename":shutil.move }

	def run_cline(self,app,args):
		request = 'python3 '+app+' '+args
		return True\
			if os.path.exists(app) and (os.system(request) == 0)\
				else False

	def run_module(self,app,args):
		m_class = app(*args)
		return True if m_class.run() == None else False

	def run_command(self,data):
		for cmd,args in data.items():
			if not cmd in self.command.keys():
				return False
			try: self.command[cmd](*args)
			except: return False
		return True

class Process:
	def __init__(self,j_path):
		super().__init__()
		self.jdata = JsonParser(j_path).run()
		self.case_module={
			True:Executor().run_module,
			False:Executor().run_cline,
		}

	def run(self,):
		for it,i in self.jdata.items():
			if not self.case_module[i[0]]( i[1][0],i[1][1] ):
				print('Error in',it) ; return False
			if i[2] != None and Executor().run_command(i[2]) == False:
				print('Error in',i[2]) ; return False
		return True

def main():
	if len(sys.argv)!=2:
		return
	return Process(sys.argv[1]).run()

if __name__ == "__main__":
	main()
