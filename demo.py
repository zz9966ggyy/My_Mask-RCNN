import cv2
import numpy as np
from mrcnn.config import Config
# 定义随机颜色函数
def random_colors(N):
    np.random.seed(1)
    colors=[tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors
 
def apply_mask(image,mask,color,alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

##########################

##########################
 
def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
 
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
 
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
 
    return image

 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background +1 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
 
if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    import samples.coco as coco
    
    # We use a K80 GPU with 24GB memory, which can fit 3 images.
    batch_size = 3
 
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = os.getcwd()    # 根目录的地址
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")   #原始视频存放的文件夹
    VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save3")   #每一帧图片所存放的文件夹
    COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"shapes20220509T2214/mask_rcnn_shapes_0010.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
 
  #  class InferenceConfig(Config):
   #     GPU_COUNT = 1
    #    IMAGES_PER_GPU = batch_size
    
    config = InferenceConfig()
    config.display()
 
    model = modellib.MaskRCNN( mode="inference", model_dir=MODEL_DIR, config=config )
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = ['BG',"pig"]
 
    capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'pigs.mp4'))  #这里是输入视频的文件名
    try:
        if not os.path.exists(VIDEO_SAVE_DIR):
            os.makedirs(VIDEO_SAVE_DIR)
    except OSError:
        print ('Error: Creating directory of data')
    frames = []
    frame_count = 0
    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break
        
        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)
        print('frame_count :{0}'.format(frame_count))
        if len(frames) == batch_size:
            results = model.detect(frames, verbose=0)
            print('Predicted')
            for i, item in enumerate(zip(frames, results)):
                frame = item[0]
                r = item[1]
                name = '{0}.jpg'.format(frame_count + i - batch_size)
                name = os.path.join(VIDEO_SAVE_DIR, name)
                frame = visualize.display_instances(
                    name,frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                )
               
                #cv2.imwrite(name, frame)

                
                print('writing to file:{0}'.format(name))
            # Clear the frames array to start the next batch
            frames = []
 
    capture.release()

    