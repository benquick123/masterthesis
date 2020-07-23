import pickle
import requests
import urllib

from skimage import io

import numpy as np

if __name__ == "__main__":
    wnids = ['n07942152',
            'n02472293',
            'n09918248',
            'n02975212',
            'n03544360',
            'n04146050',
            'n04105893',
            'n03528100',
            'n04453666',
            'n10787470',
            'n04359589',
            'n03841666',
            'n02993546',
            'n04345028',
            'n03365991',
            'n14974264',
            'n03259505',
            'n06359193',
            'n06267145',
            'n09437454']
    
    classes = ['people',
               'homo, man, human being, human',
               'child, kid',
               'case, display case, showcase, vitrine',
               'house',
               'school, schoolhouse',
               'room',
               'home, nursing home, rest home',
               'top',
               'woman, adult female',
               'support',
               'office, business office',
               'center, centre',
               'study',
               'floor, level, storey, story',
               'paper',
               'dwelling, home, domicile, abode, habitation, dwelling house',
               'web site, website, internet site, site',
               'newspaper, paper',
               'slope, incline, side']
    
    url = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid='
    
    X = []
    y = []
    for i, (wnid, cla) in enumerate(zip(wnids, classes)):
        image_urls = requests.get(url + wnid).text.split("\r\n")
        
        print("Downloading class %d: %s" % (i, cla))
        
        for j, image_url in enumerate(image_urls):
            print(j, "/", len(image_urls), ", # images:", len(X), end="\r")
            try:
                image_data = io.imread(image_url)
                X.append(image_data)
                y.append(i)
            except:
                pass
        print()
            
    X = np.array(X)
    y = np.array(y)
    
    pickle.dump((X, y), open("/opt/workspace/host_storage_hdd/imagenet_popular20", "wb"))
            