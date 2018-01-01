import pandas as pd
from bs4 import BeautifulSoup
import os
import urllib
import urllib2
import traceback
import robotparser

def main():
    
    websites = pd.read_csv('data/websites.csv')
    websites['l_uri'] = 'http://' + websites['url'].str.lower()
    

    count = 0
    for index, website in websites.iterrows():
        try:
            url = website['l_uri']
            name = website['name'].replace(' ','_').replace('/','-')
            cat = website['type']
            base = os.path.join('data/websites/', cat, name+'/')

            # filepath = os.path.join(base, name+'.txt')
            # if os.path.isfile(filepath): # already exists
            #     print('Already have:'+filepath)
            #     continue    
            # elif 'yellowpages.com' in url:
            #     continue

            text, media_url = deep_scrape(url, 3)

            if(len(text)) == 0:
                print ('continue on '+url)
                count += 1
                continue
            else:
                print('got '+url)

            if not os.path.exists(base):
                os.makedirs(base)

            with open(base+name+'.txt','wb') as output_file:
                output_file.write(text) 

            for link in media_url:
                print link, index
                index = link[8:].index('/') + 9
                name = link[index:].replace('/', '-')
                ignore_404(link, base+name)
        except:
            print ('Bad url '+url)
            print(traceback.format_exc())

def ignore_404(link, path):
    con = urllib.urlopen(link)
    print con.getcode(), link
    if con.getcode() not in [404, 401, 429, 503]:
        urllib.urlretrieve(link, path)

def filter_text(text):
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def grab_text_from_url(url):
    con = urllib.urlopen(url)
    # req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"}) 
    # con = urllib2.urlopen(req)

    # return empty if connection is lost
    code = str(con.getcode())
    if code != '200':
        print 'did not work because of this code '+code
        return '',[],[]

    html = con.read()
    soup = BeautifulSoup(html, "lxml")
    # kill all script and style elements
    for script in soup(["style", "script"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()

    text = filter_text(text)
    print text

    links = [] 
    for link in soup.find_all('a', href=True):
        links.append(link['href'])

    image_links = []
    for link in soup.findAll('img'):
        if 'src' in link:
            image_links.append(link['src'])

    return text.encode('utf-8').lower(), links,  image_links

def filter_URL(link, base_url, rp, checkRobots=True):
    if checkRobots and rp.can_fetch('*',link) == False:
        link = None
        return link

    # if no robots.txt
    if link.startswith('#') or link == '/':
        link = None
    elif not link.startswith('http'):
        if base_url.endswith('/') or link.startswith('/'):
            link = base_url+link
        else:
            link = base_url+'/'+link
    elif not link.startswith(base_url):
        link = None
    return link

def read_robot(url):
    rp = robotparser.RobotFileParser()
    useRobots = True

    try:
        rp.set_url(url+'/robots.txt')
    except:
        useRobots = False

    rp.read()
    return rp, useRobots

def deep_scrape(base_url, max_depth):
    all_text = ''
    todo_url = set()
    done_url = set()
    media_url = set()

    todo_url.add(base_url)

    rp, useRobots = read_robot(base_url)

    count = 0
    tries = 0
    while len(todo_url) > 0:
        tries += 1
        if tries > 200:
            break
        url = todo_url.pop()
        print(str(count)+':length todo '+str(len(todo_url))+' '+url)

        text, links, img_links = grab_text_from_url(url)

        # if there is no text, skip to next url
        if len(text) == 0:
            continue

        done_url.add(url)
        all_text += ' '+text
        count += 1

        if count > max_depth:
            break

        for link in img_links:
            link = filter_URL(link, base_url, rp, useRobots)
            if link != None:
                media_url.add(link)

        for u_link in links:
            link = filter_URL(u_link.lower(), base_url, rp, useRobots)
            if not link:
                continue
            if link.endswith('.jpg') == True \
                or link.endswith('.pdf') == True \
                or link.endswith('.png') == True \
                or link.endswith('.gif'):
                media_url.add(link)
            if link not in done_url:
                print('adding link '+link)
                todo_url.add(link)
    
    return all_text , media_url

if __name__ == "__main__":
    main()
