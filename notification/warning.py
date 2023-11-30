import pymysql
from linebot import LineBotApi
from linebot.models import FlexSendMessage
from datetime import datetime, timedelta
import time  # Import the time module for delays

# Database connection parameters
db_host = "209.97.174.12"
db_port = 3306
db_user = "root"
db_password = "6u&h^j=U0w)bc[f"
db_name = "flask_db"

# LINE Bot parameters
line_channel_access_token = "Replace with your API"
line_user_id = "Replace with your API"  

check_interval = 20

while True:
    try:
        print("Checking for updates...")
        # Create a connection to the MySQL database
        connection = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
        )

        # Create a cursor object to interact with the database
        cursor = connection.cursor()

        # Check for updates in the relay_status table
        cursor.execute("SELECT MAX(created_at) FROM relay_status")
        result = cursor.fetchone()

        if result and result[0] is not None:
            latest_created_at = result[0]
            # Calculate the current time
            current_time = datetime.now()

            # Calculate the time difference between the current time and the timestamp from the database
            time_difference = current_time - latest_created_at

            # Check if the time difference is less than 1 minute (60 seconds)
            if time_difference.total_seconds() < 60:
                print("Sending Locker Open Alert...")
                # Create a Flex Message
                flex_message = {
                    "type": "flex",
                    "altText": "Flex Message",
                    "contents": {
                        "type": "bubble",
                        "hero": {
                            "type": "image",
                            "url": "https://cdn-icons-png.flaticon.com/512/6360/6360268.png",
                            "size": "full",
                            "aspectRatio": "20:13"
                        },
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "⚠️ Locker Open ⚠️",
                                    "weight": "bold",
                                    "size": "xl"
                                },
                                {
                                    "type": "text",
                                    "text": "Your door locker is still open. Please check!",
                                    "wrap": True
                                }
                            ]
                        }
                    }
                }

                # Send the Flex Message to LINE
                line_bot_api = LineBotApi(line_channel_access_token)
                line_bot_api.push_message(line_user_id, FlexSendMessage(alt_text="Locker Open Alert", contents=flex_message["contents"]))
                print("Locker Open Alert sent!")

        # Close the cursor and the database connection
        cursor.close()
        connection.close()
        print("Waiting for the next check...")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Delay for the specified interval before the next check
    time.sleep(check_interval)
