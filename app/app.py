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


cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="jobintosh",
    database="flask_db"
)
mycursor = mydb.cursor()

current_directory = os.getcwd()
print("Current working directory:", current_directory)

CASCADE_CLASSIF_PATH = "app/resources/haarcascade_frontalface_default.xml"

FACE_CLASSIF_PATH = "classifier.xml"

IMAGE_TRAIN_COUNT = 100


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


# def generate_dataset(number_person, image_folder_path):
#     # receive the
#     # cap = cv2.VideoCapture(0)

#     mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
#     row = mycursor.fetchone()
#     lastid = row[0]

#     img_id = lastid
#     max_imgid = img_id + 100
#     count_img = 0

#     while True:
#         # ret, img = cap.read()
#         if face_cropped(img) is not None:
#             count_img += 1
#             img_id += 1
#             face = cv2.resize(face_cropped(img), (200, 200))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#             file_name_path = "dataset/" + \
#                 number_person+"." + str(img_id) + ".jpg"
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(count_img), (50, 50),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#             mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
#                                 ('{}', '{}')""".format(img_id, number_person))
#             mydb.commit()

#             frame = cv2.imencode('.jpg', face)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#             if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
#                 break
#                 cap.release()
#                 cv2.destroyAllWindows()


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

    # Train the classifier and save
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

                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute(
                        "insert into accs_hist (accs_date, accs_prsn) values('" + str(date.today()) + "', '" + pnbr + "')")
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
            continue  # Skip frames that are not successfully captured.

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
            "select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
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

            # Verify the hashed password
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

        # Check if the passwords match
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
                # Hash the password before storing it (you should use a stronger hashing method)
                password_hash = hashlib.sha256(password.encode()).hexdigest()

                # Insert user data into the database
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

        # Check if the email exists in the database
        cursor = mydb.cursor()
        cursor.execute(
            "SELECT id, username FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            # Generate a temporary password
            temp_password = ''.join(random.choices(
                string.ascii_letters + string.digits, k=12))
            hashed_temp_password = hashlib.sha256(
                temp_password.encode()).hexdigest()

            # Update the user's password in the database with the temporary password
            cursor.execute(
                "UPDATE users SET password_hash=%s WHERE id=%s", (hashed_temp_password, user[0]))
            mydb.commit()

            # Send an email with the temporary password
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

    # Use an SMTP server to send the email
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
            "select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
        data = mycursor.fetchall()
        username = session['username']
        return render_template('index.html', data=data, username=username)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'GET' or request.method == 'POST':
        session.pop('user_id', None)
        flash('Logout successful.', 'success')
        return redirect(url_for('login'))
    else:
        # Handle other HTTP methods if needed
        # Return a 405 Method Not Allowed status for unsupported methods
        return 'Method Not Allowed', 405


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOGIN OUTOUT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/addprsn')
def addprsn():
    if 'user_id' in session:
        mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
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

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


# @app.route('/vidfeed_dataset/<nbr>')
# def vidfeed_dataset(nbr):
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    if 'user_id' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                         "  from accs_hist a "
                         "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                         " where a.accs_date = curdate() "
                         " order by 1 desc")
        data = mycursor.fetchall()
        print("Data from fr_page:", data)  # Add this line to print data
        return render_template('fr_page.html', data=data)
    else:
        return 'You are not logged in. <a href="/login">Login</a>'


@app.route('/countTodayScan')
def countTodayScan():
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="jobintosh",
            database="flask_db"
        )
        mycursor = mydb.cursor()

        mycursor.execute("select count(*) "
                         "  from accs_hist "
                         " where accs_date = curdate() ")
        row = mycursor.fetchone()
        rowcount = row[0]
        # Add this line to print row count
        print("Row count from countTodayScan:", rowcount)
        return jsonify({'rowcount': rowcount})
    except Exception as e:
        logger.error(f"Error in countTodayScan: {str(e)}")
        # Return an error response with a status code
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    try:
        mydb = mysql.connector.connect(
            host="mariadb",
            user="root",
            passwd="jobintosh",
            database="flask_db"
        )
        mycursor = mydb.cursor()

        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                         "  from accs_hist a "
                         "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                         " where a.accs_date = curdate() "
                         " order by 1 desc")
        data = mycursor.fetchall()
        print("Data from loadData:", data)  # Add this line to print data
        return jsonify(response=data)
    except Exception as e:
        logger.error(f"Error in loadData: {str(e)}")
        # Return an error response with a status code
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/edit/<int:person_id>', methods=['GET', 'POST'])
def edit(person_id):
    if 'user_id' in session:
        if request.method == 'POST':
            new_name = request.form['name']
            new_skill = request.form['locker']

            cur = mydb.cursor()
            cur.execute("UPDATE prs_mstr SET prs_name=%s, prs_skill=%s WHERE prs_nbr=%s",
                        (new_name, new_skill, person_id))
            mydb.commit()

            flash('Personnel data updated successfully!', 'success')
            return redirect(url_for('home'))

        cur = mydb.cursor()
        cur.execute(
            "SELECT prs_name, prs_skill FROM prs_mstr WHERE prs_nbr=%s", (person_id,))
        data = cur.fetchone()
        return render_template('edit.html', data=data, person_id=person_id)
    else:
        return redirect(url_for('login'))


@app.route('/delete/<int:person_id>', methods=['GET', 'POST'])
def delete_person(person_id):
    if 'user_id' in session:
        if request.method == 'POST':
            # Delete the personnel record from the database
            cursor = mydb.cursor()  # Use mydb.cursor() instead of mydb.connection.cursor()
            cursor.execute(
                "DELETE FROM prs_mstr WHERE prs_nbr = %s", (person_id,))
            mydb.commit()  # Use mydb.commit() to save the changes
            cursor.close()

            flash('Personnel record deleted successfully!', 'success')
            return redirect(url_for('home'))

        return render_template('delete_confirmation.html', person_id=person_id)
    else:
        return redirect(url_for('login'))


######################################################
######################################################
######################################################
##################### VERSION 2 ######################
######################################################
######################################################
######################################################

def find_face(img_path):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIF_PATH)

    # Load the image from the given path
    img = cv2.imread(img_path)
    if img is None:
        return []

    # Convert the image to grayscale for face detection
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
        return

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
                label_img(imgread, "detect", 1, save_result.x, save_result.y, save_result.w, save_result.h)
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

    face_dir = os.path.join("facecrop", str(person_id))
    print(f"face train dir: {face_dir}")
    
    faces = []
    ids = []

    i = 0
    for image_path in os.listdir(face_dir):
        try:
            print(f"iter count: {i}")

            image_path = os.path.join(face_dir, image_path)
            print(f"train image: {image_path}")

            # save_result = generate_dataset_v2(
            #     person_id,
            #     image_path
            # )
            # if save_result is None:
            #     print(f"face not fount in image: {image_path}")
            #     continue

            img = Image.open(image_path).convert('L')
            imageNp = np.array(img, 'uint8')
            # id = int(os.path.split(face_cropped_image_path)[1].split(".")[1])

            faces.append(imageNp)
            ids.append(person_id)
        except Exception as e:
            print(f"cannot use image: {image_path}")
            print(e)
        finally:
            i += 1
            print("")

    ids = np.array(ids)

    # Train the classifier and save

    print("cv2 train start:")
    # print(faces)
    # print(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
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

            # Return the image as a Flask response
            # return Response(result.imgbytes, mimetype='image/jpeg')

            base64_image = base64.b64encode(result.image).decode()

            return jsonify({"message": "done", "label": result.label, "image64": base64_image})
        else:
            return jsonify({"error": "No image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# v = face_cropped_from_image_path("/workspaces/hjob-face-recognition/elon.jpg")
# print(v)


# recog_test = recognize_face_internal(
#     "dataset/122/9608dc71-f35a-4bb9-8c7f-3bc8017f71e9.jpeg")
# print(recog_test)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
