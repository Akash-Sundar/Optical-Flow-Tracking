import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os
import copy
import argparse

from optical_flow import *

def objectTracking(numberOfObjects, rawVideo):
    """
    Description: Generate and save tracking video
    Input:
        rawVideo: Raw video file name, String
    Instruction: Please feel free to use cv.selectROI() to manually select bounding box
    """
    cap = cv2.VideoCapture(rawVideo)
    imgs = []
    frame_cnt = 0 

    # number of objects to track
    F = numberOfObjects

    # Initialize video writer for tracking video
    trackVideo = 'results/Output_Easy.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(ret)
        if not ret: break
        frame_copy = copy.deepcopy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        print("frame:", frame_cnt)
        
        if frame_cnt == 1:
            bbox = np.zeros( ( F, 2, 2 ) )
            # Manually selecting objects on the first frame
            for f in range(F):
                x, y, w, h = np.int32(cv2.selectROI("roi", frame_copy, fromCenter=False))
                cv2.destroyAllWindows()
                bbox[f] = np.array( [ (x, y), (x + w, y + h) ] )
                
            features = getFeatures(frame, bbox)
            frame_old = frame.copy()

        else:
            new_features = estimateAllTranslation(features, frame_old, frame)
            features, bbox = applyGeometricTransformation(features, new_features, bbox)
            frame_old = frame.copy()
            vis = frame.copy()

            
        """ 
        TODO: Plot feature points and bounding boxes on vis
        """

        for f in range(F):
            cv2.rectangle( frame_copy, tuple( bbox[f, 0].astype(np.int32) ), tuple( bbox[f, 1].astype(np.int32) ), (0, 0, 255), thickness=2 )
            for feature in features[f]:
                cv2.circle( frame_copy, tuple(feature.astype(np.int32)), 2, (0,255,0), thickness=-1 )
        
        imgs.append(img_as_ubyte(frame_copy))

        writer.write(frame_copy)

        cv2.imshow('Track Video', frame_copy)

        if cv2.waitKey(30) & 0xff == ord('q'): break
        
        # save the video every 20 frames
        if frame_cnt % 20 == 0 or frame_cnt > 200 and frame_cnt % 10 == 0:
            imageio.mimsave('results/{}.gif'.format(frame_cnt), imgs)

    # Release video reader and video writer
    cv2.destroyAllWindows()
    cap.release()
    writer.release()

    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOfObjs', type=int, required=False)
    args = parser.parse_args()

    numberOfObjects = args.numOfObjs
    if not numberOfObjects:
        numberOfObjects = 3

    rawVideo = "test_videos/Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(numberOfObjects, rawVideo)