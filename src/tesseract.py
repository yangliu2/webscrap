from glob import glob
import os
import re

dataFolder = './val_new/*/*/'
# dataFolder = r"./val_new/83T/JITTERS_CAFE_OF_PITTSFIELD_LLC/"
# dataFolder = './val_new/83U/*/'
print dataFolder

filelist = glob(dataFolder + '*.pdf')
#filelist = glob('../../tesseract_test/*.pdf')

for f in filelist:
	print f
	png_exists = len(glob(f + '*.png')) > 0
	
	if png_exists:
		print 'png file already exists!'
		
	else:	
		f_escaped = re.escape(f)
	
		try:
	        	cstring = 'convert -density 600 -scale 2000x2000 -background white ' + f_escaped +' -colorspace Gray ' + f_escaped + '.png'
	        	os.system(cstring)
	    	#except syntaxerror:
		except Exception as e:
			errors = open('errors.txt', 'w')
			errors.write('Error on converting to png: ' + e)
			errors.close()

filelist = glob(dataFolder + '*.png')

#filelist = glob('../../tesseract_test/*.png')

for f in filelist:
	print f
	jpg_exists = len(glob(f + '*.png.jpg')) > 0
	
	if jpg_exists:
		print 'jpg file already exists!'
		
	else:		
		f_escaped = re.escape(f)
	
		try:
			cstring = 'convert -density 600 -scale 2000x2000 -background white -flatten ' + f_escaped + ' -colorspace Gray ' + f_escaped +'.jpg'
			os.system(cstring)
		except Exception as e: 
			errors = open('errors.txt', 'w')
			errors.write('Error on converting to jpg: ' + e)
			errors.close()

filelist = glob(dataFolder + '*.jpg')
#filelist = glob('../../tesseract_test/*.jpg')

for f in filelist:
	print f
	txt_exists = len(glob(f + '*.txt')) > 0
	
	if txt_exists:
		print 'txt file already exists!'
		
	else:			
		f_escaped = re.escape(f)
	
		try:
			cstring = 'tesseract --oem 2 ' + f_escaped + ' -l eng ' + f_escaped
			os.system(cstring)
		except Exception as e:
			errors = open('errors.txt', 'w')
			errors.write('Error on converting to txt: ' + e)
			errors.close()


#FILES=data/*/*/*.pdf
#for f in $FILES
#do
#        convert -density 600 -scale 2000x2000 -background white $f -colorspace Gray $f.png
#        convert -density 600 -scale 2000x2000 -background white -flatten $f.png -colorspace Gray $f.jpg
#        tesseract --oem 2 $f.jpg -l eng $f

#done


