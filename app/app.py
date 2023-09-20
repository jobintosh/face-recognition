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

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Set the log level to ERROR or lower

# Define a log file handler
handler = logging.FileHandler('error.log')  # Change the file name and path as needed
handler.setLevel(logging.ERROR)

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

 
cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="mariadb",
    user="root",
    passwd="jobintosh",
    database="flask_db"
)
mycursor = mydb.cursor()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("/resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "/dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
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
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

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

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
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
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "/resources/haarcascade_frontalface_default.xml")
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

        img = recognize(img, clf, faceCascade)

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
        mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
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
        cursor.execute("SELECT id, username, email, password_hash FROM users WHERE username=%s OR email=%s", (username_or_email, username_or_email))
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
            cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Username already in use. Please choose a different username.', 'error')
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
        cursor.execute("SELECT id, username FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            # Generate a temporary password
            temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            hashed_temp_password = hashlib.sha256(temp_password.encode()).hexdigest()

            # Update the user's password in the database with the temporary password
            cursor.execute("UPDATE users SET password_hash=%s WHERE id=%s", (hashed_temp_password, user[0]))
            mydb.commit()

            # Send an email with the temporary password
            send_password_reset_email(email, temp_password)

            flash('An email with instructions to reset your password has been sent.', 'success')
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
        smtp_server.sendmail('thetharathorn@gmail.com', [email], msg.as_string())
        smtp_server.quit()
    except Exception as e:
        print(f'Error sending email: {str(e)}')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOGIN REGISTER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/index') 
def index():
     if 'user_id' in session:
        mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
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
        return 'Method Not Allowed', 405  # Return a 405 Method Not Allowed status for unsupported methods


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
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
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
            host="mariadb",
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
        print("Row count from countTodayScan:", rowcount)  # Add this line to print row count
        return jsonify({'rowcount': rowcount})
    except Exception as e:
        logger.error(f"Error in countTodayScan: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500  # Return an error response with a status code

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
        return jsonify({'error': 'An error occurred'}), 500  # Return an error response with a status code
    
@app.route('/edit/<int:person_id>', methods=['GET', 'POST'])
def edit(person_id):
    if 'user_id' in session:
        if request.method == 'POST':
            new_name = request.form['name']
            new_skill = request.form['locker']

            cur = mydb.cursor()
            cur.execute("UPDATE prs_mstr SET prs_name=%s, prs_skill=%s WHERE prs_nbr=%s", (new_name, new_skill, person_id))
            mydb.commit()

            flash('Personnel data updated successfully!', 'success')
            return redirect(url_for('home'))

        cur = mydb.cursor()
        cur.execute("SELECT prs_name, prs_skill FROM prs_mstr WHERE prs_nbr=%s", (person_id,))
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
            cursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (person_id,))
            mydb.commit()  # Use mydb.commit() to save the changes
            cursor.close()

            flash('Personnel record deleted successfully!', 'success')
            return redirect(url_for('home'))

        return render_template('delete_confirmation.html', person_id=person_id)
    else:
        return redirect(url_for('login'))
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
