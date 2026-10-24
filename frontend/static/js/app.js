const camera = document.querySelector("#camera");
const cameraFrame = document.querySelector("#camera-frame");
const cameraButton = document.querySelector("#camera-button");
const stopButton = document.querySelector("#stop-button");
const liveIndicator = document.querySelector("#live-indicator");
const modelStatus = document.querySelector("#model-status");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");
const imageInput = document.querySelector("#image-input");
const cameraHelp = document.querySelector("#camera-help");

let cameraStream;
let captureTimer;
let cameraTimeout;
let isSendingFrame = false;
let modelAvailable = false;
let modelError = "The digits model is not available on the server.";
let cameraRequestId = 0;

function showStatus(message, isError = false) {
  modelStatus.textContent = message;
  modelStatus.classList.toggle("is-error", isError);
}

function stopCamera() {
  cameraRequestId += 1;
  window.clearTimeout(captureTimer);
  window.clearTimeout(cameraTimeout);
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
  }
  camera.srcObject = null;
  cameraFrame.classList.remove("is-active");
  liveIndicator.hidden = true;
  cameraButton.disabled = false;
  cameraButton.textContent = "Enable camera";
  stopButton.disabled = true;
  stopButton.textContent = "Stop camera";
}

function cameraAccessMessage() {
  const hostname = window.location.hostname;
  if (["localhost", "127.0.0.1", "::1"].includes(hostname)) {
    return "This browser cannot access the camera on this page. Use a current browser and allow camera access for localhost.";
  }
  return `This page is at http://${hostname}, and browsers block camera access on network addresses over HTTP. On this laptop open http://127.0.0.1:5000. To use a phone, the app must be served from a trusted HTTPS address.`;
}

function waitForPreview() {
  if (camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && camera.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      camera.removeEventListener("loadeddata", onLoaded);
      reject(new Error("The camera opened but did not provide a video picture. Check that it is not covered or disabled, then try again."));
    }, 10000);
    function onLoaded() {
      window.clearTimeout(timer);
      resolve();
    }
    camera.addEventListener("loadeddata", onLoaded, { once: true });
  });
}

async function getCameraStream(requestId) {
  let lastError;
  const constraints = [
    { video: { facingMode: { ideal: "environment" } }, audio: false },
    { video: true, audio: false },
  ];

  for (const constraint of constraints) {
    if (requestId !== cameraRequestId) {
      throw new DOMException("Camera startup was cancelled.", "AbortError");
    }
    try {
      return await navigator.mediaDevices.getUserMedia(constraint);
    } catch (error) {
      lastError = error;
      if (requestId !== cameraRequestId) {
        throw new DOMException("Camera startup was cancelled.", "AbortError");
      }
      if (error.name === "NotAllowedError" || error.name === "SecurityError") {
        throw error;
      }
    }
  }

  throw lastError;
}

async function requestPrediction(blob) {
  const formData = new FormData();
  formData.append("image", blob, "camera-frame.jpg");

  const response = await fetch("/api/predict", { method: "POST", body: formData });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Recognition failed.");
  }
  prediction.textContent = data.label;
  confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%. ${data.message || ""}`;
}

function scheduleFrame() {
  captureTimer = window.setTimeout(captureFrame, 900);
}

async function captureFrame() {
  if (!cameraStream || isSendingFrame) return;
  if (camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || camera.videoWidth === 0) {
    scheduleFrame();
    return;
  }

  isSendingFrame = true;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    canvas.getContext("2d").drawImage(camera, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
    if (!blob) throw new Error("Could not capture an image from the camera.");
    await requestPrediction(blob);
  } catch (error) {
    prediction.textContent = "Recognition unavailable";
    confidence.textContent = error.message;
    if (error.message.includes("No trained") || error.message.includes("TensorFlow")) {
      modelAvailable = false;
      modelError = error.message;
      showStatus(error.message, true);
      stopCamera();
      return;
    }
  } finally {
    isSendingFrame = false;
  }
  scheduleFrame();
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    const message = cameraAccessMessage();
    prediction.textContent = "Camera needs a secure page";
    confidence.textContent = message;
    showStatus(message, true);
    return;
  }

  const requestId = ++cameraRequestId;
  cameraButton.disabled = true;
  cameraButton.textContent = "Starting camera…";
  stopButton.disabled = false;
  stopButton.textContent = "Cancel";
  prediction.textContent = "Starting camera…";
  confidence.textContent = "Allow camera access in your browser if prompted.";

  const streamRequest = getCameraStream(requestId);
  streamRequest.then((stream) => {
    if (requestId !== cameraRequestId) {
      stream.getTracks().forEach((track) => track.stop());
    }
  }, () => {});

  try {
    cameraStream = await Promise.race([
      streamRequest,
      new Promise((_, reject) => {
        cameraTimeout = window.setTimeout(
          () => reject(new DOMException("Camera start timed out.", "TimeoutError")),
          15000,
        );
      }),
    ]);
    window.clearTimeout(cameraTimeout);
    if (requestId !== cameraRequestId) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = undefined;
      return;
    }

    const stream = cameraStream;
    camera.srcObject = cameraStream;
    await camera.play();
    await waitForPreview();
    if (requestId !== cameraRequestId) return;
    cameraFrame.classList.add("is-active");
    liveIndicator.hidden = false;
    stopButton.disabled = false;
    stopButton.textContent = "Stop camera";
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if (cameraStream !== stream) return;
      stopCamera();
      prediction.textContent = "Camera disconnected";
      confidence.textContent = "The camera stopped. Check its connection or permission and try again.";
    }, { once: true });
    if (modelAvailable) {
      prediction.textContent = "Looking for a sign…";
      confidence.textContent = "Hold one digit sign clearly in the camera view.";
      scheduleFrame();
    } else {
      prediction.textContent = "Camera is ready";
      confidence.textContent = modelError;
    }
  } catch (error) {
    window.clearTimeout(cameraTimeout);
    if (requestId !== cameraRequestId) return;
    stopCamera();
    prediction.textContent = "Camera not started";
    const message = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow access in your browser settings and try again."
      : error.name === "NotFoundError"
        ? "No camera was found on this device. Connect or enable a camera, then try again."
        : error.name === "TimeoutError"
          ? "Camera startup timed out. Check the browser's camera permission prompt, then try again."
          : error.name === "AbortError"
            ? "Camera startup was cancelled by the browser. Close other camera apps and try again."
        : error.name === "NotReadableError" || /device in use/i.test(error.message)
          ? "The camera is busy or unavailable. Close other apps or browser tabs using it (such as Teams, Zoom, or Camera), then try again."
          : `Could not start the camera: ${error.message}`;
    confidence.textContent = message;
    showStatus(message, true);
    cameraButton.textContent = "Try camera again";
  }
}

cameraButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => {
  const wasStarting = cameraButton.disabled;
  stopCamera();
  if (wasStarting) {
    prediction.textContent = "Camera start cancelled";
    confidence.textContent = "Click Enable camera to try again.";
  } else {
    prediction.textContent = "Camera stopped";
    confidence.textContent = "Click Enable camera when you are ready.";
  }
});
imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  prediction.textContent = "Processing image…";
  confidence.textContent = "";
  try {
    await requestPrediction(file);
  } catch (error) {
    prediction.textContent = "Recognition unavailable";
    confidence.textContent = error.message;
  }
});

window.addEventListener("pagehide", stopCamera);

if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
  const message = cameraAccessMessage();
  cameraHelp.textContent = message;
  showStatus(message, true);
  cameraButton.disabled = true;
  cameraButton.textContent = "Camera requires HTTPS";
}

fetch("/api/health")
  .then((response) => {
    if (!response.ok) throw new Error("Could not check backend status.");
    return response.json();
  })
  .then((data) => {
    const digitsModel = data.models?.digits;
    modelAvailable = Boolean(digitsModel?.available);
    modelError = digitsModel?.error || "The digits model is not available on the server.";
    if (window.isSecureContext && navigator.mediaDevices?.getUserMedia) {
      showStatus(
        modelAvailable
          ? "Digits model ready. Camera frames are processed for predictions."
          : modelError,
        !modelAvailable,
      );
    }
    if (modelAvailable && cameraStream) {
      prediction.textContent = "Looking for a sign…";
      confidence.textContent = "Hold one digit sign clearly in the camera view.";
      scheduleFrame();
    } else if (!modelAvailable && cameraStream) {
      prediction.textContent = "Model unavailable";
      confidence.textContent = modelError;
    }
  })
  .catch((error) => showStatus(error.message, true));

if (window.isSecureContext && navigator.mediaDevices?.getUserMedia) {
  startCamera();
}
