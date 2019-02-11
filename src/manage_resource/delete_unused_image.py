#! python3
# -*- coding: utf-8 -*-

import os
from manage_resource.util import Pysql

diseases = ["blight/", "powdery/", "rust/"]
pysql = Pysql()

for disease in diseases:
    file_list = os.listdir("../../data/images/"+disease)
    print(len(file_list))
    for file in file_list:
        pysql.update(disease[:-1], is_deleted=0, file_name=file)

pysql.close()
