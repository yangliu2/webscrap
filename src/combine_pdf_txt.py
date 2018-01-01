from glob import glob
import os
import re
import shutil

dataFolder = './val_new/'
#dataFolder = './test/'
files = glob(dataFolder + '*/*/*')

#print files
unique = set()
for i in files:
	dir_path = os.path.dirname(os.path.realpath(i))
	unique.add(dir_path)
print 'unique dir_path',unique

for i in list(unique):
	# dir_path = os.path.dirname(os.path.realpath(i))
	outputFile = os.path.join(i, 'output.txt')
	print 'output file', outputFile
	textFiles = glob(i+'/*.txt')
	with open(outputFile, 'w') as output:
		for inputFile in textFiles:
			if inputFile == outputFile:
				print 'output file skipped'
				continue
			with open(inputFile, 'r') as readfile:
				print 'input file', inputFile
				shutil.copyfileobj(readfile, output)

#for dir in os.listdir(dataFolder):
#	for sdir in os.listdir(os.path.join(dataFolder, dir)):
#		filelist = glob(os.path.join(dataFolder, dir, sdir) + '*.txt')
#		outfilename = os.path.join(dataFolder, dir, sdir) + 'combined_pdf_txt.txt'
#		print dir
#		with open(outfilename, 'wb') as outfile:
#		    for filename in glob('*.txt'):
#		        with open(filename, 'rb') as readfile:
#		            shutil.copyfileobj(readfile, outfilename + 'combined_pdf_txt.txt')
#		for f in filelist:
#			with open(os.path.join(dataFolder, dir, sdir) + 'combined_pdf_txt.txt', 'w') as outfile:
#			    for fname in filenames:
#			        with open(fname) as infile:
#			            for line in infile:
#			                outfile.write(line)
