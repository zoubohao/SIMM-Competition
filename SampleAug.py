from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision as tv
import cv2

# def mixUp(image1, image2):
#     im1Re = tv.transforms.ToTensor()(image1)
#     im2Re = tv.transforms.ToTensor()(image2)
#     return tv.transforms.ToPILImage()(0.4 * im1Re + 0.6 * im2Re)

# def microAug(pilResize):
#     img = tv.transforms.ToTensor()(pilResize)
#     #print(img.shape)
#     circle = cv2.circle((np.ones([img.shape[1], img.shape[2]]) * 255).astype(np.uint8),
#                         (np.random.randint(img.shape[2]//2 - 40, img.shape[2]//2 + 40),
#                          np.random.randint(img.shape[1]//2 - 40, img.shape[1]//2 + 40)),
#                         np.random.randint(img.shape[2]//2 - 40, img.shape[2]//2 + 40),
#                         (0, 0, 0),-1)
#     mask = circle - 255
#     img = np.multiply(img, mask)
#     return tv.transforms.ToPILImage()(img)

# def hairsAug(img):
#     n_hairs = np.random.randint(15, 20)
#     height, width, _ = img.shape  # target image width and height
#     hair_images = [im for im in os.listdir("./hair") if 'png' in im]
#
#     for _ in range(n_hairs):
#         hair = cv2.imread(os.path.join("./hair", np.random.choice(hair_images)))
#         hair = cv2.flip(hair, np.random.choice([-1, 0, 1]))
#         hair = cv2.rotate(hair, np.random.choice([0, 1, 2]))
#
#         h_height, h_width, _ = hair.shape  # hair image width and height
#         roi_ho = np.random.randint(0, img.shape[0] - hair.shape[0])
#         roi_wo = np.random.randint(0, img.shape[1] - hair.shape[1])
#         roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]
#
#         img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
#         ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
#         mask_inv = cv2.bitwise_not(mask)
#         img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
#         hair_fg = cv2.bitwise_and(hair, hair, mask=mask)
#
#         dst = cv2.add(img_bg, hair_fg)
#         img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
#     img = cv2.resize(img, dsize=(int(408 * 1.118), int(272 * 1.118)), interpolation=cv2.INTER_LANCZOS4)
#     return img

### Random crop ----> 342, this is in training process.
### Center crop  ---> 387
### Resize -----> 430

def CutMix(image1, image2):
    imageW = 408
    imageH = 272
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
    if_pos = False
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

    centerCrop = tv.transforms.CenterCrop([272, 408])
    resize = tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)])
    randomCrop = tv.transforms.RandomCrop([272, 408])

    for i, imgName in enumerate(imgsNames):
        thisAnatom = anatom[i]
        print(i)
        if if_pos:
            imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
            resizePIL = resize(imgPIL)
            ### ori 272, 408
            centerCropPIL = centerCrop(resizePIL)
            centerCropPIL.save(os.path.join(savePath, imgName + ".jpg"))
            augNames.append(imgName)
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])
            ### VF
            imgVF = tv.transforms.RandomVerticalFlip(p=1.)(centerCropPIL)
            imgVF.save(os.path.join(savePath, imgName + "_VerFlip.jpg"))
            augNames.append(imgName + "_VerFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])
            ### HF
            imgHF = tv.transforms.RandomHorizontalFlip(p=1.)(centerCropPIL)
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
            imgVFHF = transformations(centerCropPIL)
            imgVFHF.save(os.path.join(savePath, imgName + "_VerHorFlip.jpg"))
            augNames.append(imgName + "_VerHorFlip")
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### random crop
            op = [tv.transforms.Compose([
                tv.transforms.RandomVerticalFlip(p=1.),
                tv.transforms.RandomHorizontalFlip(p=1.)
                                        ]),
                tv.transforms.RandomVerticalFlip(p=1.),
                tv.transforms.RandomHorizontalFlip(p=1.),
                tv.transforms.CenterCrop([int(272 * 1.118), int(408 * 1.118)])
            ]
            for t in range(31):
                randCropPIL = randomCrop(op[np.random.randint(0,4)](resizePIL))
                randCropPIL.save(os.path.join(savePath, imgName + "_RandomCrop" + str(t) + ".jpg"))
                augNames.append(imgName + "_RandomCrop" + str(t))
                augSex.append(sexs[i])
                augAge.append(ages[i])
                augAtom.append(anatom[i])

            cutMixList = [centerCropPIL, imgHF, imgVF, imgVFHF]
            ### cut mix
            for t in range(35):
                j = np.random.randint(0, len(imgsNames))
                while j == i:
                    j = np.random.randint(0, len(imgsNames))
                currentImageName = imgsNames[j]
                currentSex = sexs[j]
                currentAge = ages[j]
                currentAnatom = anatom[j]
                currentImgPIL = Image.open(os.path.join("./train", currentImageName + ".jpg")).convert(
                    "RGB")
                cutMixImage = CutMix(cutMixList[np.random.randint(0,4)],  centerCrop(resize(currentImgPIL)))
                cutMixImage.save(
                    os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + str(t) + ".jpg"))
                augNames.append(imgName + "_" + currentImageName + "_CutUp" + str(t))
                augSex.append(currentSex)
                augAge.append(currentAge)
                augAtom.append(currentAnatom)

        else:

            imgPIL = Image.open(os.path.join("./train", imgName + ".jpg")).convert("RGB")
            resizePIL = resize(imgPIL)
            ### ori
            centerCropPIL = centerCrop(resizePIL)
            centerCropPIL.save(os.path.join(savePath, imgName + ".jpg"))
            augNames.append(imgName)
            augSex.append(sexs[i])
            augAge.append(ages[i])
            augAtom.append(anatom[i])

            ### cut mix
            if np.random.rand(1) <= 0.3:
                j = np.random.randint(0, len(imgsNames))
                while j == i:
                    j = np.random.randint(0, len(imgsNames))
                currentImageName = imgsNames[j]
                currentSex = sexs[j]
                currentAge = ages[j]
                currentAnatom = anatom[j]
                currentImgPIL = Image.open(os.path.join("./train", currentImageName + ".jpg")).convert(
                    "RGB")
                cutMixImage = CutMix(centerCropPIL, centerCrop(resize(currentImgPIL)))
                cutMixImage.save(
                    os.path.join(savePath, imgName + "_" + currentImageName + "_CutUp" + ".jpg"))
                augNames.append(imgName + "_" + currentImageName + "_CutUp")
                augSex.append(currentSex)
                augAge.append(currentAge)
                augAtom.append(currentAnatom)


    if if_pos:
        targets = [1 for _ in range(len(augNames))]
        data = {"image_name": augNames, "sex": augSex,
                "age_approx": augAge, "anatom_site_general_challenge": augAtom, "target": targets}
        dataFrame = pd.DataFrame(data=data).to_csv("./CSVFile/" + csvName, index=False)
    else:
        targets = [0 for _ in range(len(augNames))]
        data = {"image_name": augNames, "sex": augSex,
                "age_approx": augAge, "anatom_site_general_challenge": augAtom, "target": targets}
        dataFrame = pd.DataFrame(data=data).to_csv("./CSVFile/" + csvName, index=False)







