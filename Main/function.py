def detect_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect_faces(img_rgb)
    detection = detections[0]
    x, y, w, h = detection["box"]
    detected_face = img[int(y):int(y+h), int(x):int(x+w)]
    return detected_face

def preprocess_face(img, target_size=(112,112)):
    img = cv2.imread(img)
    img = detect_face(img)
    img = cv2.resize(img, target_size)
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    return img_pixels

def img_to_encoding(path):
    img = preprocess_face(path)
    return model.predict(img)[0]

database = {}
arr_ = os.listdir('./images')
for i in filter(lambda x: '.jpg' in x, arr_):    
    # print(i.split('.')[0])
    name = i.split('.')[0]
    path = './images/' + i
    database[name] = img_to_encoding(path)

verification_threshhold = 3.0

def EuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def who_is_it(image_path, database):
    encoding = img_to_encoding(image_path)
    status = ""
    min_dist = 1000
    for (name, db_enc) in database.items():
        dist = EuclideanDistance(encoding, db_enc)
        if min_dist > dist:
            min_dist = dist
            identity = name
