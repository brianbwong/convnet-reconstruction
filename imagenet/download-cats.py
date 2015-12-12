import urllib

with open('cat-urls.txt', 'r') as f:
    content = f.read()

counter = 13
for url in content.split('\n')[13:]:
    img_filename = 'cats/' + str(counter) + '.jpg'
    with open(img_filename, 'wb') as f:
        try:
            f.write(urllib.urlopen(url).read())
        except: 
            print 'Failed to find image'
    counter += 1