iport os
from unidecode import unidecode
import sys


def parse_newline(text):
    return text.replace('\\n', ' ')

def convert_unicode(text):
	# convert unicode in string into unicode
	text = text.decode('unicode_escape') 
	return text

def parse_symbols(text):
	text = text.replace('"', '')
	return unidecode(text)

def parse_dash(text):
	text = text.replace('--', '-')
	return text

def parse_text(text):
	text = convert_unicode(text)
	# text = parse_newline(text)

	text = parse_symbols(text)
	text = parse_dash(text)
	print text
	return text

def read_file(path):
	text = ''
	with open(path, 'rb') as file:
		for line in file:
			text += line
	return text

def save_file(path, file):
	with open(path, 'wb') as output:
		output.write(file)

def parse_OCR(filename):
	raw_text = read_file(filename)
	parsed_text = parse_text(raw_text)
	
	output_file = os.path.basename(filename).split('.')[0] + '_parsed.txt'
	save_file(output_file, parsed_text)

def main(filename):
	parse_OCR(filename)


if __name__ == '__main__':
	main(sys.argv[1])
