# STEP 1: import modules
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2: create inference object(instance) 
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3: load data
img1 = cv2.imread('ri1.jpg')
img2 = cv2.imread('ri2.jpg')

# STEP 4: inference
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# STEP 5: post processing
# STEP 5-1: save result image
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# STEP 5-2: face recognition
# then print all-to-all face similarity

feat_face1=faces1[0].normed_embedding
feat_face2=faces2[0].normed_embedding

feat_face1 = np.array(feat_face1, dtype=np.float32)
feat_face2 = np.array(feat_face2, dtype=np.float32)

sims = np.dot(feat_face1, feat_face2.T)
print(sims)
