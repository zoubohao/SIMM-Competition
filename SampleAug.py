from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np
import torchvision as tv

def mixUp(image1, image2):

    imageW = 3096
    imageH = 2048

    im1Re = cv2.resize(image1,dsize=(imageW, imageH),interpolation=cv2.INTER_LANCZOS4)
    im2Re = cv2.resize(image2, dsize=(imageW, imageH),interpolation=cv2.INTER_LANCZOS4)

    return 0.5 * im1Re + 0.5 * im2Re

def CutMix(image1, image2):
    imageW = 3096
    imageH = 2048
    ##
    im1 = tv.transforms.Resize([imageH, imageW])(image1)
    im2 = tv.transforms.Resize([imageH, imageW])(image2)
    la = np.random.randint(2,4)
    ##
    randx = np.random.randint(1,imageW // la * (la - 1) - 5)
    randy = np.random.randint(1,imageH // la * (la - 1) - 5)
    ##
    mask = np.zeros(shape=[1,imageH,imageW])
    mask[0,randy : (randy + imageH // la), randx : (randx + imageW // la)] = 1
    ##
    maskImg1 = tv.transforms.ToTensor()(im1) * mask
    maskImg2 = tv.transforms.ToTensor()(im2) * (1 - mask)
    ##
    addImg = (maskImg1 + maskImg2).float()
    return tv.transforms.ToPILImage()(addImg)

if __name__ == "__main__":
    dataInfor = pd.read_csv("./CSVFile/trainNeg.csv")
    savePath = "./NegTrainAug"
    if_pos = False
    csvName = "augNeg.csv"
    if_part = True

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

        thisSex = sexs[i]
        thisAge = ages[i]
        thisAnatom = anatom[i]
        ### ori
        print(i)
        im = cv2.imread(os.path.join("./train",imgName + ".jpg"))
        cv2.imwrite(os.path.join(savePath,imgName + ".jpg"),im)
        augNames.append(imgName)
        augSex.append(sexs[i])
        augAge.append(ages[i])
        augAtom.append(anatom[i])

        if if_pos:
            ### color
            b, g, r = cv2.split(im)
            cv2.imwrite(os.path.join(savePath, imgName + "BlueC.jpg"), b)
            augNames.append(imgName + "BlueC")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            cv2.imwrite(os.path.join(savePath, imgName + "RedC.jpg"), r)
            augNames.append(imgName + "RedC")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            cv2.imwrite(os.path.join(savePath, imgName + "GreenC.jpg"), g)
            augNames.append(imgName + "GreenC")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### hist
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)

            imgColorHis = cv2.merge([bH, gH, rH])
            cv2.imwrite(os.path.join(savePath, imgName + "_ColorHist.jpg"), imgColorHis)
            augNames.append(imgName + "_ColorHist")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### gray
            imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(savePath, imgName + "_Gray.jpg"), imGray)
            augNames.append(imgName + "_Gray")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### gamma
            img2 = np.power(imGray / float(np.max(imGray)), 1.875)
            cv2.imwrite(os.path.join(savePath, imgName + "_GammaMul.jpg"), img2 * 255.)
            augNames.append(imgName + "_GammaMul")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### VF
            imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
            imgVF = tv.transforms.RandomVerticalFlip(p=1.)(imgPIL)
            imgVF.save(os.path.join(savePath, imgName + "_VerFlip.jpg"))
            augNames.append(imgName + "_VerFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### HF
            imgHF = tv.transforms.RandomHorizontalFlip(p=1.)(imgPIL)
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
            imgVFHF = transformations(imgPIL)
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
                        currentImgPIL = Image.open(os.path.join("./train", currentImageName + ".jpg")).convert(
                            "RGB")
                        cutMixImage = CutMix(imgPIL, currentImgPIL)
                        cutMixImage.save(
                            os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + ".jpg"))
                        augNames.append(imgName + "_" + currentImageName + "_CutUp")
                        augSex.append(currentSex)
                        augAge.append(currentAge)
                        augAtom.append(currentAnatom)
        else:
            if i not in [0,1,2,3]:
                if np.random.rand(1) <= 0.5:
                    j = np.random.randint(0, i)
                    imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
                    currentImageName = imgsNames[j]
                    currentSex = sexs[j]
                    currentAge = ages[j]
                    currentAnatom = anatom[j]
                    currentImgPIL = Image.open(os.path.join("./train", currentImageName + ".jpg")).convert(
                        "RGB")
                    cutMixImage = CutMix(imgPIL, currentImgPIL)
                    cutMixImage.save(
                        os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + ".jpg"))
                    augNames.append(imgName + "_" + currentImageName + "_CutUp")
                    augSex.append(currentSex)
                    augAge.append(currentAge)
                    augAtom.append(currentAnatom)


    targets = [1 for _ in range(len(augNames))]
    data = {"image_name": augNames, "sex" : augSex,
            "age_approx" : augAge, "anatom_site_general_challenge" : augAtom, "target" : targets}
    dataFrame = pd.DataFrame(data=data).to_csv("./CSVFile/" + csvName,index=False)






