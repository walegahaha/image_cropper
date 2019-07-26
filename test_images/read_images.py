import requests

with open('image_url.txt') as f:
    for i, url in enumerate(f.readlines()):
        #url = "https://img.alicdn.com/tfscom/O1CN01eoIiKe29edsloSR6W_!!0-rate.jpg"
        r = requests.get(url[:-1]) 
        with open("%d.jpg" %i, "wb") as code:
            code.write(r.content)

