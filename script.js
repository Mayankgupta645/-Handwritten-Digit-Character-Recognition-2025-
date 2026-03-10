function predict() {
  const fileInput = document.getElementById("imageInput");
  const result = document.getElementById("result");

  if (!fileInput.files.length) {
    result.innerText = "⚠️ Please upload an image first";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

 
  result.innerHTML = `
    <div class="scanner">
      <div class="line"></div>
      <p>AI SCANNING IMAGE...</p>
    </div>
  `;

  fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    result.innerHTML = `
      <h2>✅ SCAN COMPLETE</h2>
      <p>Digit Detected: <b>${data.digit}</b></p>
      <p>Confidence: <b>${data.confidence}%</b></p>
    `;
  })
  .catch(err => {
    result.innerText = "❌ Prediction failed. Check backend.";
    console.error(err);
  });
}
