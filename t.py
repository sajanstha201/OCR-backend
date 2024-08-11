import layoutparser as lp
import cv2
from PIL import Image

def usePubLayNet(i):
    publaynet = "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config"
    pb_model= lp.PaddleDetectionLayoutModel(publaynet)
    pb_layout = pb_model.detect(i)
    pb_blocks = pb_layout._blocks
    print(len(pb_blocks))

    # print(f'{len(pb_blocks)} pb_blocks: \n')
    # for item in pb_blocks:
    #     print(item)
    bboxes = []
    for item in pb_blocks:
        print(item.type, item)
        bbox = item.block
        bboxes.append(((int(bbox.x_1),int(bbox.y_1),int(bbox.x_2),int(bbox.y_2)), item.type, item.score))
        
    return bboxes


path = 'page_96.png'
image = cv2.imread(path)
# image = Image.open(path).convert('RGB')
playnet = usePubLayNet(image)



for item in playnet:
    cv2.rectangle(
        image,
        (int(item[0][0]),int(item[0][1])),
        (int(item[0][2]), int(item[0][3])),
        (255,0,0),
        3
    )
    
cv2.imwrite('result.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
