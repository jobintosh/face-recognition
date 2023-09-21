
console.log("uploadURL", uploadURL);

const videoElement = document.getElementById('camera-stream');
var mediaStream
var isDone = false;

// Function to start the camera and return a promise
const startCamera = () => {
    return new Promise(async (resolve, reject) => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true })
            videoElement.srcObject = mediaStream
            resolve('Camera started')
        } catch (error) {
            reject(error)
        }
    })
}

// Function to stop the camera
const stopCamera = () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop())
        videoElement.srcObject = null
    }
}


function captureFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg').split(',')[1];; // Convert to base64 image data
    // console.log(imageData);

    fetch(uploadURL, {
        method: 'POST',
        body: JSON.stringify({ image: imageData }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then((response) => response.text())
        .then((responseText) => {
            console.log('Server response:', responseText)

            if (responseText.includes("done") && isDone == false){
                isDone = true;
                window.location.replace(trainURL);
            }
        })
        .catch((error) => {
            console.error('Error sending data to server:', error)
        })
}

// Start capturing frames when the video is playing
videoElement.addEventListener('play', function () {
    captureFrame();
});

// Optionally, you can stop capturing frames when the video is paused or ended
videoElement.addEventListener('pause', function () {
    cancelAnimationFrame(captureFrame);
});
videoElement.addEventListener('ended', function () {
    cancelAnimationFrame(captureFrame);
});

startCamera()

setInterval(() => {
    captureFrame()
}, 10);
