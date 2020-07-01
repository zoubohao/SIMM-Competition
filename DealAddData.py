import os
import pandas as pd
from shutil import copyfile
import re

### name, meta.clinical.age_approx, meta.clinical.anatom_site_general, meta.clinical.sex
def main(walk_path, copy_path):
    dataMap = {}
    isic_names = list()
    age_approx = list()
    anatom = list()
    sex = list()
    ### csv regex
    regex = re.compile(r"^.*\.csv$")
    for root, dirs, files in os.walk(walk_path):
        for name in files:
            print(os.path.join(root, name))
            texts = regex.match(name)
            if texts is not None:
                currentDF = pd.read_csv(os.path.join(root, name))
                isic_names = isic_names + list(currentDF["name"])
                age_approx = age_approx + list(currentDF["meta.clinical.age_approx"])
                anatom = anatom + list(currentDF["meta.clinical.anatom_site_general"])
                sex = sex + list(currentDF["meta.clinical.sex"])
            else:
                if ".txt" not in name:
                    ### notice that there are some png pic in this data set.
                    ### you should convert it to jpg handel.
                    copyfile(os.path.join(root, name), os.path.join(copy_path, name))

    dataMap["image_name"] = isic_names
    dataMap["sex"] = sex
    dataMap["age_approx"] = age_approx
    dataMap["anatom_site_general_challenge"] = anatom
    dataMap["target"] = [1 for _ in range(len(isic_names))]
    pd.DataFrame(dataMap).to_csv("./CSVFile/Additional_Pos.csv",index=False)


if __name__ == "__main__":
    main("./AddtionalData","./testCopy")
    trainPos = pd.read_csv("./CSVFile/trainPos.csv")
    addPos = pd.read_csv("./CSVFile/Additional_Pos.csv")








