import urllib2

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from config import path_to_training_data, class_name

for i in range(2):
    cat_url = "https://cataas.com/cat"


    for y in range(2500):
        url = ""
        if i == 0:
            url = cat_url
        else:
            contents = urllib2.urlopen("https://dog.ceo/api/breeds/image/random").read()

            dog_url = eval(contents).get('message').replace('\/', "/")
            url = dog_url
        print(str(i) + "_" + str(y))

        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)


        response = session.get(url)
        # response = requests.get(url)

        file = open(path_to_training_data + "/" + class_name + str(i) + "_" + str(y) + ".png", "wb")
        file.write(response.content)
        file.close()

    # for y in range(500):