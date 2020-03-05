import sys,os

def check():
	if not os.path.exists(sys.argv[1])\
	or not os.path.exists(sys.argv[2])\
	or not os.path.exists(sys.argv[3])\
	or not os.path.exists(sys.argv[4]):
		print('Error: Input directory / file not found')
		raise FileNotFoundError

def main():
	if len(sys.argv)!=5:
		print('Error: Invalid number of parameters')
		raise ValueError
	check()
	#csv train test img ttrain test
	os.system('python3 generate_tfrecord.py\
		--csv_input='+sys.argv[1]+'\
		--output_path=train.record\
		--image_dir='+sys.argv[3])
	os.system('python3 generate_tfrecord.py\
			--csv_input='+sys.argv[2]+'\
			--output_path=test.record\
			--image_dir='+sys.argv[4])

if __name__ == "__main__":
	main()