import requests
import time
import json

# Replace with your Line Messaging API Channel Access Token
access_token = 'Replace with your API'

# Line Messaging API URL for sending messages
line_url = 'https://api.line.me/v2/bot/message/push'

# Recipient's Line user or group ID
recipient_id = 'Replace with your API'

# Fetch JSON data from your server
server_url = 'http://209.97.174.12/loadData'


# Interval between each fetch and message send (in seconds)
fetch_interval = 10  # Change this to your desired interval

# Variable to store the previous JSON data
previous_data = None


try:
    while True:
        # Make a GET request to fetch JSON data
        response = requests.get(server_url)

        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            # Parse the JSON response from your server
            server_data = response.json()

            # Check if 'response' is in the JSON data and it's a non-empty list
            if 'response' in server_data and isinstance(server_data['response'], list) and server_data['response']:
                scan_data = server_data['response'][0]  # Assuming you want the first entry

                # Check if scan_data has at least 5 elements
                if len(scan_data) >= 5:
                    # Compare the new data with the previous data
                    if server_data != previous_data:
                        # Update the previous data with the new data
                        previous_data = server_data

                        # Create a Flex Message using the extracted data
                        flex_message = {
                            "type": "flex",
                            "altText": "your locker open!",
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
                                            "text": f"🧑🏻‍💻: {scan_data[2]}",
                                            "wrap": True
                                        },
                                        {
                                            "type": "text",
                                            "text": f"🧳: {scan_data[3]}",
                                            "wrap": True
                                        },
                                        {
                                            "type": "text",
                                            "text": f"⏱️: {scan_data[4]}",
                                            "wrap": True
                                        }
                                    ]
                                }
                            }
                        }

                        # JSON data for the Line message
                        message_data = {
                            "to": recipient_id,
                            "messages": [flex_message]
                        }

                        # Set the HTTP headers for the Line API request
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": "Bearer " + access_token
                        }

                        # Send the Line message with the Flex Message payload
                        line_response = requests.post(line_url, json=message_data, headers=headers)

                        # Check the Line API response
                        if line_response.status_code == 200:
                            print("Flex Message sent successfully.")
                        else:
                            print("Error sending Flex Message:", line_response.status_code)
                    else:
                        print("No new data. Skipping message send.")
                else:
                    print("Not enough elements in scan_data.")
            else:
                print("Invalid or empty 'response' in JSON data from the server.")
        else:
            print(f"Error fetching data from server. Status code: {response.status_code}")

        # Sleep for the specified interval before the next iteration
        time.sleep(fetch_interval)

except KeyboardInterrupt:
    # Handle the case where the user manually stops the program (Ctrl+C)
    print("Program stopped by user.")
except Exception as e:
    print("An error occurred:", str(e))
