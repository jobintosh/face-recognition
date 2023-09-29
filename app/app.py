import base64
import uuid
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
import logging
import hashlib
import pymysql
import smtplib
from email.mime.text import MIMEText
import random
import string
import cv2
from datetime import datetime
import stripe

stripe.api_key = 'sk_test_51NutLPBGGzOWH6WKvYHArctigy1hgkbnJKYslyXf4qDpm6gsbD1H7vA3YFVisTrkJIpve1DiV9yruIZ5QZ8dc4KS00dh18YMQB'


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Set the log level to ERROR or lower

# Define a log file handler
# Change the file name and path as needed
handler = logging.FileHandler('error.log')
handler.setLevel(logging.ERROR)

# Define a log format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Add the application attribute required by Gunicorn
application = app

cnt = 0
pause_cnt = 0
justscanned = False

# load from env MYSQL_HOST first, if not found, use default value 'localhost'
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')

mydb = mysql.connector.connect(
    host="mariadb",
    user="root",
    passwd="6u&h^j=U0w)bc[f",
    database="flask_db"
)
mycursor = mydb.cursor()

current_directory = os.getcwd()
print("Current working directory:", current_directory)

CASCADE_CLASSIF_PATH = "app/resources/haarcascade_frontalface_default.xml"

FACE_CLASSIF_PATH = "classifier.xml"

IMAGE_TRAIN_COUNT = 300


class RecognizeResult:
    label: str
    confidence: float
    image: any


class SaveResult(RecognizeResult):
    save_path: str
    x: int
    y: int
    h: int
    w: int

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

face_classifier = cv2.CascadeClassifier(CASCADE_CLASSIF_PATH)


def face_cropped(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # scaling factor=1.3
    # Minimum neighbor = 5

    if faces == ():
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(
            gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            gray_face = gray_image[y:y + h, x:x + w]
            id, pred = clf.predict(gray_face)
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40),
                              (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255),
                              cv2.FILLED)

                mycursor.execute("select a.img_person, b.full_name, b.locker_no "
                                 "  from img_dataset a "
                                 "  left join personnel_data b on a.img_person = b.personnel_id "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute(
                        "insert into activity_log (accs_date, accs_prsn) values('" + str(date.today()) + "', '" + pnbr + "')")
                    mydb.commit()

                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(
                        img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10,
                               (255, 255, 0), "Face", clf)
        return img

    face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIF_PATH)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()

        if not ret:
            continue

        img = recognize(img, clf, face_cascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOGIN REGISTER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/')
def home():
    if 'user_id' in session:
        mycursor.execute(
            "select personnel_id, full_name, locker_no, active, added from personnel_data")
        data = mycursor.fetchall()
        username = session['username']
        return render_template('index.html', data=data, username=username)
    else:
        return 'You are not logged in. <a href="/login">Login</a>'
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        password = request.form['password']
        

        cursor = mydb.cursor()
        cursor.execute("SELECT id, username, email, password_hash FROM users WHERE username=%s OR email=%s",
                       (username_or_email, username_or_email))
        user = cursor.fetchone()

        if user:
            user_id, db_username, db_email, db_password_hash = user

            input_password_hash = hashlib.sha256(password.encode()).hexdigest()
            if input_password_hash == db_password_hash:
                session['user_id'] = user_id
                session['username'] = db_username
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Incorrect password. Please try again.', 'error')
        else:
            flash('Login failed. Please check your credentials.', 'error')

    return render_template('login.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username_or_email = request.form['username_or_email']
#         password = request.form['password']
#         recaptcha_response = request.form['g-recaptcha-response']  # รับข้อมูล reCAPTCHA

#         recaptcha_secret_key = '6Leck1IoAAAAAA3kDnwU_ovbJvY8jXn7K5Ln5x-Q'  # รหัสลับ reCAPTCHA (ให้เปลี่ยนเป็นรหัสที่คุณได้รับ)
#         recaptcha_url = 'https://www.google.com/recaptcha/api/siteverify'
#         recaptcha_data = {
#             'secret': recaptcha_secret_key,
#             'response': recaptcha_response
#         }

#         response = requests.post(recaptcha_url, data=recaptcha_data)
#         recaptcha_result = response.json()

#         if recaptcha_result['success']:
#             # reCAPTCHA ยืนยันสำเร็จ
#             cursor = mydb.cursor()
#             cursor.execute("SELECT id, username, email, password_hash FROM users WHERE username=%s OR email=%s",
#                            (username_or_email, username_or_email))
#             user = cursor.fetchone()

#             if user:
#                 user_id, db_username, db_email, db_password_hash = user

#                 input_password_hash = hashlib.sha256(password.encode()).hexdigest()
#                 if input_password_hash == db_password_hash:
#                     session['user_id'] = user_id
#                     session['username'] = db_username
#                     flash('Login successful!', 'success')
#                     return redirect(url_for('home'))
#                 else:
#                     flash('Incorrect password. Please try again.', 'error')
#             else:
#                 flash('Login failed. Please check your credentials.', 'error')
#         else:
#             # reCAPTCHA ไม่ยืนยัน
#             flash('reCAPTCHA verification failed. Please try again.', 'error')

#     return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phonenumber = request.form['phonenumber']
        password_confirm = request.form['password_confirm']

        if password != password_confirm:
            flash('Passwords do not match. Please try again.', 'error')
        else:
            cursor = mydb.cursor()
            cursor.execute(
                "SELECT id FROM users WHERE username=%s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash(
                    'Username already in use. Please choose a different username.', 'error')
            else:
                password_hash = hashlib.sha256(password.encode()).hexdigest()

                cursor.execute("INSERT INTO users (username, password_hash, firstname, lastname, email, phonenumber) VALUES (%s, %s, %s, %s, %s, %s)",
                               (username, password_hash, firstname, lastname, email, phonenumber))
                mydb.commit()

                flash('Registration successful. You can now log in.', 'success')
                return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        cursor = mydb.cursor()
        cursor.execute(
            "SELECT id, username FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            temp_password = ''.join(random.choices(
                string.ascii_letters + string.digits, k=12))
            hashed_temp_password = hashlib.sha256(
                temp_password.encode()).hexdigest()

            cursor.execute(
                "UPDATE users SET password_hash=%s WHERE id=%s", (hashed_temp_password, user[0]))
            mydb.commit()

            send_password_reset_email(email, temp_password)

            flash(
                'An email with instructions to reset your password has been sent.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email address not found. Please try again.', 'error')

    return render_template('forgot_password.html')


def send_password_reset_email(email, temp_password):
    msg = MIMEText(f'Your temporary password is: {temp_password}')
    msg['Subject'] = 'Password Reset'
    msg['From'] = 'thetharathorn@gmail.com'
    msg['To'] = email

    try:
        smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
        smtp_server.starttls()
        smtp_server.login('thetharathorn@gmail.com', 'urep nusy qtkj vvpq')
        smtp_server.sendmail('thetharathorn@gmail.com',
                             [email], msg.as_string())
        smtp_server.quit()
    except Exception as e:
        print(f'Error sending email: {str(e)}')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOGIN REGISTER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/index')
def index():
    if 'user_id' in session:
        mycursor.execute(
            "select personnel_id, full_name, locker_no, active, added from personnel_data")
        data = mycursor.fetchall()
        username = session['username']
        return render_template('index.html', data=data, username=username)
    else:
        return 'You are not logged in. <a href="/login">Login</a>'


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'GET' or request.method == 'POST':
        session.pop('user_id', None)
        flash('Logout successful.', 'success')
        return redirect(url_for('login'))
    else:
        return 'Method Not Allowed', 405


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOGIN OUTOUT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/addprsn')
def addprsn():
    if 'user_id' in session:
        mycursor.execute("select ifnull(max(personnel_id) + 1, 101) from personnel_data")
        row = mycursor.fetchone()
        nbr = row[0]
        # print(int(nbr))
        return render_template('addprsn.html', newnbr=int(nbr))
    else:
        return 'You are not logged in. <a href="/login">Login</a>'


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `personnel_data` (`personnel_id`, `full_name`, `locker_no`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    if 'user_id' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.full_name, b.locker_no, a.accs_added "
                         "  from activity_log a "
                         "  left join personnel_data b on a.accs_prsn = b.personnel_id "
                         " where a.accs_date = curdate() "
                         " order by 1 desc")
        data = mycursor.fetchall()
        print("Data from fr_page:", data)
        return render_template('fr_page.html', data=data)
    else:
        return 'You are not logged in. <a href="/login">Login</a>'


@app.route('/countTodayScan')
def countTodayScan():
    try:
        mydb = mysql.connector.connect(
            host="mariadb",
            user="root",
            passwd="6u&h^j=U0w)bc[f",
            database="flask_db"
        )
        mycursor = mydb.cursor()

        mycursor.execute("select count(*) "
                         "  from activity_log "
                         " where accs_date = curdate() ")
        row = mycursor.fetchone()
        rowcount = row[0]
        print("Row count from countTodayScan:", rowcount)
        return jsonify({'rowcount': rowcount})
    except Exception as e:
        logger.error(f"Error in countTodayScan: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    try:
        mydb = mysql.connector.connect(
            host="mariadb",
            user="root",
            passwd="6u&h^j=U0w)bc[f",
            database="flask_db"
        )
        mycursor = mydb.cursor()

        mycursor.execute("select a.accs_id, a.accs_prsn, b.full_name, b.locker_no, date_format(a.accs_added, '%H:%i:%s') "
                         "  from activity_log a "
                         "  left join personnel_data b on a.accs_prsn = b.personnel_id "
                         " where a.accs_date = curdate() "
                         " order by 1 desc")
        data = mycursor.fetchall()
        print("Data from loadData:", data) 
        return jsonify(response=data)
    except Exception as e:
        logger.error(f"Error in loadData: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/edit/<int:person_id>', methods=['GET', 'POST'])
def edit(person_id):
    if 'user_id' in session:
        if request.method == 'POST':
            new_name = request.form['name']
            new_skill = request.form['locker']

            cur = mydb.cursor()
            cur.execute("UPDATE personnel_data SET full_name=%s, locker_no=%s WHERE personnel_id=%s",
                        (new_name, new_skill, person_id))
            mydb.commit()

            flash('Personnel data updated successfully!', 'success')
            return redirect(url_for('home'))

        cur = mydb.cursor()
        cur.execute(
            "SELECT full_name, locker_no FROM personnel_data WHERE personnel_id=%s", (person_id,))
        data = cur.fetchone()
        return render_template('edit.html', data=data, person_id=person_id)
    else:
        return redirect(url_for('login'))


@app.route('/delete/<int:person_id>', methods=['GET', 'POST'])
def delete_person(person_id):
    if 'user_id' in session:
        if request.method == 'POST':
            cursor = mydb.cursor()
            cursor.execute(
                "DELETE FROM personnel_data WHERE personnel_id = %s", (person_id,))
            mydb.commit()
            cursor.close()

            flash('Personnel record deleted successfully!', 'success')
            return redirect(url_for('home'))

        return render_template('delete_confirmation.html', person_id=person_id)
    else:
        return redirect(url_for('login'))



########################################## VERSION 2 ###########################################################


def find_face(img_path):
    face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIF_PATH)

    img = cv2.imread(img_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces


def save_image_base64(image_base64, folder_path):
    # Generate a unique filename
    unique_filename = str(uuid.uuid4()) + ".jpeg"
    image_path = os.path.join(folder_path, unique_filename)

    # Decode the base64 image data and save it to the file
    with open(image_path, "wb") as img_file:
        img_data = base64.b64decode(image_base64)
        img_file.write(img_data)

    return unique_filename


def label_img(img, label, confidence, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label_text = f"{label} ({confidence:.2f})"
    cv2.putText(
        img,
        label_text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0),
        2,
    )


def recognize_face_from_img_path(img_path):
    img = cv2.imread(img_path)
    faces = find_face(img_path)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return None

    # Save the input image with recognized faces
    output_image_path = 'test.jpeg'

    # Load a pre-trained face recognition model (you need to replace this with your model)
    # Example:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    # You should train this recognizer on a dataset of known faces

    # Recognize the detected faces (you need to implement this based on your recognizer)
    recognized_faces = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces:
        # Crop the detected face from the image
        face_roi = gray[y:y+h, x:x+w]

        # Perform face recognition (replace this with your recognition code)
        # Example:
        label, confidence = recognizer.predict(face_roi)

        # Add recognized faces to the list
        recognized_faces.append({
            'bbox': (x, y, w, h),
            'label': label,  # Replace with your recognized label or name
            'confidence': confidence  # Replace with your recognition confidence score
        })

    label = None
    confidence = 0
    for face in recognized_faces:
        x, y, w, h = face['bbox']
        label = face['label']
        confidence = face['confidence']

        print(f"Face: {label}, Confidence: {confidence}")

        # label the image
        label_img(img, label, confidence, x, y, w, h)

        # break after first face found!
        break

    # Save the image with rectangles drawn around recognized faces
    cv2.imwrite(output_image_path, img)
    print(f"Image with recognized faces saved as {output_image_path}")

    result = RecognizeResult()

    result.label = label
    result.confidence = confidence

    _, image_data = cv2.imencode('.jpg', img)
    result.image = image_data

    return result


def generate_dataset_v2(person_id, img_path):
    faces = find_face(img_path)

    # No faces detected in the image
    if len(faces) == 0:
        return None

    # Assume only one face is detected (you can modify this if needed)
    (x, y, w, h) = faces[0]

    img = cv2.imread(img_path)

    # Crop the detected face from the image
    face_cropped_img = img[y:y+h, x:x+w]

    if face_cropped_img is None:
        return None

    face = cv2.resize(face_cropped_img, (200, 200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face_cropped_dir = os.path.join("facecrop", str(person_id))
    os.makedirs(face_cropped_dir, exist_ok=True)

    filename = os.path.basename(img_path)

    save_path = os.path.join(face_cropped_dir, filename)
    cv2.imwrite(save_path, face)

    result = SaveResult()

    result.image = face
    result.label = person_id
    result.save_path = save_path
    result.x = x
    result.y = y
    result.h = h
    result.w = w

    return result


@app.route('/upload/<int:person_id>', methods=['POST'])
def upload_image(person_id):
    try:
        data = request.json
        if 'image' in data:
            image_data_base64 = data['image']

            # create upload directory
            upload_dir_path = os.path.join("dataset", str(person_id))
            # Create the folder if it doesn't exist
            os.makedirs(upload_dir_path, exist_ok=True)

            # create face crop directory
            facecrop_dir = os.path.join("facecrop", str(person_id))
            os.makedirs(facecrop_dir, exist_ok=True)

            # Check the number of images in the folder
            image_count = len([f for f in os.listdir(facecrop_dir)])

            if image_count >= IMAGE_TRAIN_COUNT:
                # send status done
                # client side will redirect to train_face route
                return jsonify({"status": "done"})
            else:            # Save the image to the folder
                filename = save_image_base64(
                    image_data_base64,
                    upload_dir_path
                )

                file_full_path = os.path.join(upload_dir_path, filename)

                save_result = generate_dataset_v2(person_id, file_full_path)
                if save_result is None:
                    return jsonify({"message": "face not found", "filename": filename}), 200

                imgread = cv2.imread(file_full_path)
                label_img(imgread, "detect", 1, save_result.x,
                          save_result.y, save_result.w, save_result.h)
                imgencode = cv2.imencode('.jpg', imgread)[1].tobytes()
                base64_image = base64.b64encode(imgencode).decode()

                return jsonify({
                    "message": "image saved successfully",
                    "filename": filename,
                    "image_count": image_count,
                    "image64": base64_image
                }), 200
        else:
            return jsonify({"error": "No image field in request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train/<person_id>')
def train_face(person_id):
    person_id = int(person_id)
    print(f"train start for person id: {person_id}")

    face_dir = os.path.join("facecrop", str(person_id))
    print(f"face train dir: {face_dir}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    is_old_classif_exists = os.path.exists("classifier.xml")

    # if classifier.xml exists, read it
    if is_old_classif_exists is True:
        # read the existing classifier
        recognizer.read("classifier.xml")
        
    # init variable
    faces = []
    ids = []

    i = 0
    for image_path in os.listdir(face_dir):
        try:
            image_path = os.path.join(face_dir, image_path)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            flattened_img = np.array(img, 'uint8')
            
            faces.append(flattened_img)
            ids.append(person_id)
        except Exception as e:
            print(f"cannot use image: {image_path}")
            print(e)
            print("")
        finally:
            i += 1

    # convert to numpy arrays
    faces = np.array(faces)
    ids = np.array(ids)

    print("cv2 train start:")

    # Train the classifier
    if is_old_classif_exists is True:
        recognizer.update(faces, ids)
    else:
        recognizer.train(faces, ids)
    
    # save classifier
    recognizer.write("classifier.xml")
    
    print(f"trained person id: {person_id}")

    return redirect('/')


@app.route('/recognize', methods=['POST'])
def recognize_v2():
    try:
        data = request.json
        if 'image' in data:
            image_data = data['image']
            folder_path = os.path.join("recognize")

            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Save the image to the folder
            filename = save_image_base64(image_data, folder_path)

            image_full_path = os.path.join(folder_path, filename)

            result: RecognizeResult = recognize_face_from_img_path(
                image_full_path)

            if result is None:
                return jsonify({"message": "result not found"}), 400

            # Check confidence level
            print(f"confidence level: {result.confidence}")
            if result.confidence <= 40:
                return jsonify({"message": "low confidence"}), 400

            # Now, let's fetch the recognized person's label and save it in activity_log
            recognized_label = result.label

            # Insert data into the 'activity_log' table
            sql = "INSERT INTO activity_log (accs_date, accs_prsn, accs_added) VALUES (%s, %s, %s)"
            val = (datetime.today(), recognized_label, datetime.now())
            mycursor.execute(sql, val)
            mydb.commit()

            # Encode the result image and send it to detect.js
            base64_image = base64.b64encode(result.image).decode()

            return jsonify({"message": "done", "label": recognized_label, "image64": base64_image})
        else:
            return jsonify({"error": "No image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/edit_profile/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    if 'user_id' not in session:
        # User is not logged in, so redirect to the login page
        return redirect(url_for('login'))

    # Check if the logged-in user is authorized to edit this profile
    if session['user_id'] != user_id:
        # Unauthorized access, you can handle this as per your application's requirements
        return "Unauthorized access"

    # The rest of your code for editing the profile remains the same
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = mycursor.fetchone()

    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phonenumber = request.form['phonenumber']

        # Check if a new password was provided
        new_password = request.form['new_password']

        # Update user data in the database
        if new_password:
            # Hash the new password before updating it
            hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
            mycursor.execute("UPDATE users SET firstname = %s, lastname = %s, email = %s, phonenumber = %s, password_hash = %s WHERE id = %s",
                             (firstname, lastname, email, phonenumber, hashed_password, user_id))
        else:
            mycursor.execute("UPDATE users SET firstname = %s, lastname = %s, email = %s, phonenumber = %s WHERE id = %s",
                             (firstname, lastname, email, phonenumber, user_id))

        mydb.commit()
        mycursor.close()
        return redirect(url_for('index'))

    mycursor.close()
    return render_template('profile.html', user=user)

YOUR_DOMAIN = 'https://locker.jobintosh.me'

@app.route('/success')
def success():
    return render_template('success.html')

# Define the route for the cancel page
@app.route('/cancel')
def cancel():
    return render_template('cancel.html')

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                    'price': 'price_1NutemBGGzOWH6WKIywdDoNf',
                    'quantity': 1,
                },
            ],
            mode='payment',
            success_url=YOUR_DOMAIN + '/success',
            cancel_url=YOUR_DOMAIN + '/cancel',
        )
    except Exception as e:
        return str(e)

    return redirect(checkout_session.url, code=303)

@app.route('/webhook', methods=['POST'])
def webhook():
    event = None
    payload = request.data

    try:
        event = json.loads(payload)
    except:
        print('⚠️  Webhook error while parsing basic request.' + str(e))
        return jsonify(success=False)
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = request.headers.get('stripe-signature')
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, endpoint_secret
            )
        except stripe.error.SignatureVerificationError as e:
            print('⚠️  Webhook signature verification failed.' + str(e))
            return jsonify(success=False)

    # Handle the event
    if event and event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']  # contains a stripe.PaymentIntent
        print('Payment for {} succeeded'.format(payment_intent['amount']))
        # Then define and call a method to handle the successful payment intent.
        # handle_payment_intent_succeeded(payment_intent)
    elif event['type'] == 'payment_method.attached':
        payment_method = event['data']['object']  # contains a stripe.PaymentMethod
        # Then define and call a method to handle the successful attachment of a PaymentMethod.
        # handle_payment_method_attached(payment_method)
    else:
        # Unexpected event type
        print('Unhandled event type {}'.format(event['type']))

    return jsonify(success=True)
    
@app.route('/logs')
def logs():
    if 'user_id' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.full_name, b.locker_no, a.accs_added "
                         "  from activity_log a "
                         "  left join personnel_data b on a.accs_prsn = b.personnel_id "
                         " order by 1 desc")
        data = mycursor.fetchall()
        print("Data from logs:", data)
        return render_template('logs.html', data=data)
    else:
        return 'You are not logged in. <a href="/login">Login</a>'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)