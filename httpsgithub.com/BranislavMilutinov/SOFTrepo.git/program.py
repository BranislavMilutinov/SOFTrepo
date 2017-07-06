import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog

# Funkcija da prepoznamo PLAVU liniju
def fline(image_rgb):
    image_gray = np.ndarray((image_rgb.shape[0], image_rgb.shape[1])) # dimenzije slike

    for i in np.arange(0, image_rgb.shape[0]):  #piksel po piksel po dimenzijiama

        for j in np.arange(0, image_rgb.shape[1]):
            if  image_rgb[i, j, 0] >= 150 and  image_rgb[i, j, 1] < 50 and image_rgb[i, j, 2] < 50:
                image_gray[i, j] = 255  #boji u belo ono od znacaja
            else:
                image_gray[i, j] = 0   #boji u crno ono sto nam ne treba
    image_gray = image_gray.astype('uint8')


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))  #za pravougaonik koji okviri
    image_gray_dill = cv2.dilate(image_gray, kernel, iterations=2)

    return image_gray_dill #dilatacija kad hocu da zavorim prekide na liniji

# Funkcija da prepoznamo BEO broj

def fnumber(image_rgb):
    image_gray = np.ndarray((image_rgb.shape[0], image_rgb.shape[1]))


    #sa 155 definisemo belinu bele boje
    for i in np.arange(0, image_rgb.shape[0]):
        for j in np.arange(0, image_rgb.shape[1]):
            if image_rgb[i, j, 0] > 155 and image_rgb[i, j, 1] > 155 and image_rgb[i, j, 2] > 155:
                image_gray[i, j] = 255
            else:
                image_gray[i, j] = 0
    image_gray = image_gray.astype('uint8')


    image_gray_dill = cv2.dilate(image_gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)

    return image_gray_dill


# za pronalazenje lin jednacine y=kx+n

def lin_eq(x_1, y_1, x_2, y_2):
    k=0.0
    y_22 = y_2+0.000
    y_11 = y_1+0.000
    k = (y_22-y_11)/(x_2-x_1)
    n = k*(-x_1) + y_1
    return [k, n]


# Proverava preklapanje izmedju linije i tacke,
def overlap(k, n, x, y):
    y_jednacina=k*x+n #jednacina prave
    prag=0.7
    if abs(y_jednacina-y)<prag:
        return True
    else:
        return False
#sa 0.7 definisem prag, ako je lin_eq na udaljenosti manjoj od praga onda racunam taj broj

# ucitavanje baze rukopisa
clf = joblib.load("digits_cls.pkl")



output_file = open('outx.txt', 'w')
output_file.close()

for p in np.arange(0, 10):  # Treba da prodjemo kroz sve snimke koji su nam zadati

    suma = 0
    im_line = None
    k0 = 0
    n0 = 0

    capture = cv2.VideoCapture("VideosTest/video-" + str(p) + ".avi")  # Ucitavanje videa
    print "video-" + str(p) + ".avi"

    startFrame = 0
    capture.set(1, startFrame);
    while True:
        startFrame += 1
        ret, frame = capture.read()
        if not ret:
            break

        # operacije koje vrsimo nad linijom radimo samo na prvom frejmu posto se linija ne mrda
        if im_line is None:
            im_line = fline(frame)                                                                       # Liniu sa slike izvlacimo
            _, contures_line, _ = cv2.findContours(im_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # trazi se kontura te linije
            rectangles_line = [cv2.boundingRect(conture) for conture in contures_line]                                       # izvlaci se pravougaoni okvir na osnovu konture
            rectangle_line = rectangles_line[0]   # izvlacenje jedinog pravougaonika
            height = rectangle_line[3]       # visina linije
            width = rectangle_line[2]       # sirina linije
            [k0, n0] = lin_eq(rectangle_line[0], 480 - rectangle_line[1] - height, rectangle_line[0] + width, 480 - rectangle_line[1])  # trazenje jedn. prave
            line_range = (rectangle_line[0], rectangle_line[0] + width)    # max i min sirina linije


        # operacije nad brojevima
        im_number = fnumber(frame)                                                                       # brojebe izblacimo sa slike

        _, contures_num, _ = cv2.findContours(im_number.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # trazi se kontura brojeva

        rectangles_num = [cv2.boundingRect(conture) for conture in contures_num]                                       # iizvlaci se pravougaoni okvir na osnovu konture

        for rectangle in rectangles_num:  # Prolazimo kroz regione ( za brojeve )
            height_rn = rectangle[3]  # visina regiona broja
            width_rn = rectangle[2]  # sirina regiona broja

            # centar regiona broja
            x_centar = rectangle[0] + width_rn / 2
            y_centar = 480 - (rectangle[1] + height_rn / 2)


            # Formiranje pravougaonika oko naseg broja
            leng = int(height_rn * 1.2)
            point1 = int(rectangle[1] + height_rn // 2 - leng // 2)
            point2 = int(rectangle[0] + width_rn // 2 - leng // 2)

            #broj na ivici slike u gornjem levom coksu prilikom kretanja moze da baci gresku, tako da ako nije u opsegu setuje se na nulu

            if point1 < 0:
                point1 = 0
            else: point1

            if point2 < 0:
                point2 = 0
            else: point2

            reg_of_int = im_number[point1:point1 + leng, point2:point2 + leng]
            # Umanjivanje slike za znacajni region
            reg_of_int = cv2.resize(reg_of_int, (28, 28), interpolation=cv2.INTER_AREA)
            reg_of_int = cv2.dilate(reg_of_int, (3, 3))
            # Racunanje za histogram
            reg_of_int_hog_fd = hog(reg_of_int, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([reg_of_int_hog_fd], 'float64'))

            # Proveramo centar regiona broja - da li je on na jednoj od dve prave
            if  overlap(k0, n0, x_centar, y_centar) and line_range[0] < x_centar < line_range[1]:
                suma += int(nbr)

        print 'Zbir: ' + str(suma) + ' -- frame: ' + str(startFrame) +  ' ~' + str(startFrame / 40.0) + ' sec'

    # pisanje
    output_file = open('results.txt', 'a')
    output_file.write("video-" + str(p) + '\t' + str(suma) + '\n')
    output_file.close()
    capture.release()