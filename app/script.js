const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const errorDiv = document.getElementById('error');

// Handle file selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewImage.classList.add('show');
        };
        reader.readAsDataURL(file);
        
        predictBtn.disabled = false;
        resultsSection.classList.remove('show');
        errorDiv.classList.remove('show');
    }
});

// Predict image
async function predictImage() {
    const file = imageInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        errorDiv.classList.remove('show');
        loading.classList.add('show');
        predictBtn.disabled = true;

        // Determine API URL - adjust if needed
        const apiUrl = 'http://localhost:8000/predict';

        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        // Display results
        document.getElementById('predictionResult').textContent = data.class_id;
        document.getElementById('camImage').src = `data:image/png;base64,${data.cam_image}`;
        
        resultsSection.classList.add('show');
    } catch (error) {
        console.error('Error:', error);
        errorDiv.textContent = `Error: ${error.message}`;
        errorDiv.classList.add('show');
    } finally {
        loading.classList.remove('show');
        predictBtn.disabled = false;
    }
}

// Allow Enter key to predict
imageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !predictBtn.disabled) {
        predictImage();
    }
});
