import cv2
import sys
import dlib
from skimage import io
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from time import sleep

imagePath = 'test.jpg'
modelFullPath = 'output_graph.pb'
labelsFullPath = 'output_labels.txt'
size = 8
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

webcam = cv2.VideoCapture(0) 

predictor_path = 'shape_predictor_68_face_landmarks.dat'
#image_path = sys.argv[2]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def compute_EAR(vec):

	a = distance.euclidean(vec[1], vec[5])
	b = distance.euclidean(vec[2], vec[4])
	c = distance.euclidean(vec[0], vec[3])
	# compute EAR
	ear = (a + b) / (2.0 * c)

	return ear
def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None
    (_ ,img) = webcam.read()
    sleep(2)
    img=cv2.flip(img,1,0)
    
    #win = dlib.image_window()
    #win.clear_overlay()
    #win.set_image(img)
    dets = detector(img, 1)
    with open('faces.txt' , 'w') as f1:
        f1.write(str(len(dets)))
    vec = np.empty([68, 2], dtype = int)
    
    status="Not Sleeping"
    print("Number of faces detected: {}".format(len(dets)))
    p = 1 
    #q = 1
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
            
        right_ear=compute_EAR(vec[42:48])
        left_ear=compute_EAR(vec[36:42])
        if (right_ear+left_ear)/2 <0.2:
            status="sleeping"
            with open("emotion_{}.txt".format(p) , 'w') as f:
                f.write(status)
            p = p + 1


        
            
        print(status)
        #win.add_overlay(shape)
    if(status=="Not Sleeping"):
        #if not tf.gfile.Exists(imagePath):
        #   tf.logging.fatal('File does not exist %s', imagePath)
        #  return answer
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,0) #Flip to act as a mirror
        mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)
        q = 1
        x = 1
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
            #Save just the rectangle faces in SubRecFaces
            sub_face = im[y:y+h, x:x+w]

            FaceFileName = "test.jpg" #Saving the current image from the webcam for testing.
            face = "face_{}.jpg".format(x)
            cv2.imwrite(face , sub_face)
            x = x + 1
            cv2.imwrite(FaceFileName, sub_face)
        
    # Resize the image to speed up detection
    #image_data = cv2.resize(im, 64, 64)
      

   

        image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
        create_graph()

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                 {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
            max_score = 0.0
            max_string = ""
            for node_id in top_k:
                human_string = labels[node_id]
                '''
                with open("emotionN_{}.txt".format(q) , 'a') as f:
                    f.write(human_string)
                '''
                score = predictions[node_id]
                if score > max_score:
                    max_score = score
                    max_string = human_string

                print('%s (score = %.5f)' % (human_string, score))
            with open("emotion_{}.txt".format(q) , 'w') as f:
                f.write(max_string)
            q = q + 1
            answer = labels[top_k[0]]
            return answer


if __name__ == '__main__':
  while True:
    run_inference_on_image()
    sleep(4)
