import urllib
import random

n = 100
urls = []

# open image url file and download random subset (only from flickr)
# imagenet url files available at http://image-net.org/download-imageurls

with open('fall11_urls.txt') as f:
    lines = f.readlines()

    while len(urls) < n:
        url = random.choice(lines).split()[1]
        if 'flickr' not in url:
            continue

        urls.append(url)
        urllib.urlretrieve(url, 'img/%d.jpg' % len(urls))
        if len(urls) >= n:
            break
