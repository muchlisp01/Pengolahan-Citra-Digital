import os
from collections import Counter
from deepface import DeepFace

models = DeepFace.build_model('Facenet512')

def findMode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

def findFace(imageInputs, model):
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    df = DeepFace.find(img_path = imageInputs, db_path = "/Users/muchl/UIN/PCD/UAS/kumpulanPhoto", detector_backend = backends[4],\
                       model_name = models[2], model = model,  distance_metric = metrics[0],  enforce_detection = False)
    col = list(df)[-1]
    dfFinal = df[df[col] <= 0.21]
    
    if len(dfFinal) == 0:
        result = {'status': 'Unrecognize', 'name':'Unrecognize'}
    else:
        candidat_name = findMode([os.path.basename(os.path.dirname(dfFinal['identity'][i])) for i in range(len(dfFinal))])[0]
        result = {'status': 'Recognize', 'name':candidat_name}
    
    return result