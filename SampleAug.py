from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision as tv

# def mixUp(image1, image2):
#
#     imageW = int(408 * 1.118)
#     imageH = int(272 * 1.118)
#
#     im1Re = cv2.resize(image1,dsize=(imageW, imageH),interpolation=cv2.INTER_LANCZOS4)
#     im2Re = cv2.resize(image2, dsize=(imageW, imageH),interpolation=cv2.INTER_LANCZOS4)
#
#     return 0.5 * im1Re + 0.5 * im2Re

def CutMix(image1, image2):
    imageW = int(408 * 1.118)
    imageH = int(272 * 1.118)
    ##
    la = np.random.randint(2,4)
    ##
    randx = np.random.randint(1,imageW // la * (la - 1) - 5)
    randy = np.random.randint(1,imageH // la * (la - 1) - 5)
    ##
    mask = np.zeros(shape=[1,imageH,imageW])
    mask[0,randy : (randy + imageH // la), randx : (randx + imageW // la)] = 1
    ##
    maskImg1 = tv.transforms.ToTensor()(image1) * mask
    maskImg2 = tv.transforms.ToTensor()(image2) * (1 - mask)
    ##
    addImg = (maskImg1 + maskImg2).float()
    return tv.transforms.ToPILImage()(addImg)

if __name__ == "__main__":
    dataInfor = pd.read_csv("./CSVFile/trainNeg.csv")
    savePath = "./NegTrainAugResize"
    if_pos = True
    csvName = "augNeg.csv"
    if_part = False

    imgsNames = np.array(dataInfor["image_name"])
    sexs = np.array(dataInfor["sex"])
    ages = np.array(dataInfor["age_approx"])
    anatom = np.array(dataInfor["anatom_site_general_challenge"])

    print(len(imgsNames))
    print(len(sexs))
    print(len(ages))
    print(len(anatom))

    length = 28000

    if if_part:
        index = list(range(length))
        np.random.shuffle(index)
        imgsNames = imgsNames[index][0: length]
        sexs = sexs[index][0: length]
        ages = ages[index][0: length]
        anatom = anatom[index][0: length]

    print(len(imgsNames))
    print(len(sexs))
    print(len(ages))
    print(len(anatom))

    augNames = []
    augSex = []
    augAge = []
    augAtom = []

    for i, imgName in enumerate(imgsNames):
        thisAnatom = anatom[i]
        print(i)

        if if_pos:
            ### ori
            imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
            resizePIL = tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)])(imgPIL)
            resizePIL.save(os.path.join(savePath, imgName + ".jpg"))
            augNames.append(imgName)
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### VF
            imgVF = tv.transforms.RandomVerticalFlip(p=1.)(resizePIL)
            imgVF.save(os.path.join(savePath, imgName + "_VerFlip.jpg"))
            augNames.append(imgName + "_VerFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### HF
            imgHF = tv.transforms.RandomHorizontalFlip(p=1.)(resizePIL)
            imgHF.save(os.path.join(savePath, imgName + "_HorFlip.jpg"))
            augNames.append(imgName + "_HorFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### VF and HF
            transformations = tv.transforms.Compose([
                tv.transforms.RandomVerticalFlip(p=1.),
                tv.transforms.RandomHorizontalFlip(p=1.)
            ])
            imgVFHF = transformations(resizePIL)
            imgVFHF.save(os.path.join(savePath, imgName + "_VerHorFlip.jpg"))
            augNames.append(imgName + "_VerHorFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            for j in range(i):
                currentAnatom = anatom[j]
                if currentAnatom == thisAnatom:
                    if np.random.rand(1) <= 0.75:
                        currentImageName = imgsNames[j]
                        currentSex = sexs[j]
                        currentAge = ages[j]
                        currentImgPIL = Image.open(os.path.join(savePath, currentImageName + ".jpg")).convert(
                            "RGB")
                        cutMixImage = CutMix(resizePIL, currentImgPIL)
                        cutMixImage.save(
                            os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + ".jpg"))
                        augNames.append(imgName + "_" + currentImageName + "_CutUp")
                        augSex.append(currentSex)
                        augAge.append(currentAge)
                        augAtom.append(currentAnatom)
            targets = [1 for _ in range(len(augNames))]
            data = {"image_name": augNames, "sex": augSex,
                    "age_approx": augAge, "anatom_site_general_challenge": augAtom, "target": targets}
            dataFrame = pd.DataFrame(data=data).to_csv("./CSVFile/" + csvName, index=False)


        else:
            imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
            resizePIL = tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)])(imgPIL)
            resizePIL.save(os.path.join(savePath, imgName + ".jpg"))
            augNames.append(imgName)
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])
            if i not in [0,1,2,3]:
                if np.random.rand(1) <= 0.36:
                    j = np.random.randint(0, i)
                    currentImageName = imgsNames[j]
                    currentSex = sexs[j]
                    currentAge = ages[j]
                    currentAnatom = anatom[j]
                    currentImgPIL = Image.open(os.path.join(savePath, currentImageName + ".jpg")).convert(
                        "RGB")
                    cutMixImage = CutMix(resizePIL, currentImgPIL)
                    cutMixImage.save(
                        os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + ".jpg"))
                    augNames.append(imgName + "_" + currentImageName + "_CutUp")
                    augSex.append(currentSex)
                    augAge.append(currentAge)
                    augAtom.append(currentAnatom)
            targets = [0 for _ in range(len(augNames))]
            data = {"image_name": augNames, "sex": augSex,
                    "age_approx": augAge, "anatom_site_general_challenge": augAtom, "target": targets}
            dataFrame = pd.DataFrame(data=data).to_csv("./CSVFile/" + csvName, index=False)






