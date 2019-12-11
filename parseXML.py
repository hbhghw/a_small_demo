from xml.etree import ElementTree
import os


def _parse(file):
    root = ElementTree.parse(file)
    box = root.find('object').find('bndbox')
    path = root.find('path').text #修改
    xmin = box.find('xmin').text
    xmax = box.find('xmax').text
    ymin = box.find('ymin').text
    ymax = box.find('ymax').text
    return [path,xmin,xmax,ymin,ymax]

with open('FaceDB.txt','w') as f:
    root = '/home/hw/Data/two/label'
    for file in os.listdir(root):
        if file.endswith('xml'):
            ret = _parse(os.path.join(root,file))
            f.write(' '.join(ret))
            f.write('\n')
