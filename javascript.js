document.getElementById('image-form').addEventListener('submit', async function(event) {
event.preventDefault();
const prompt = document.getElementById('prompt').value;
const response = await fetch('/generate', {
method: 'POST',
headers: {
'Content-Type': 'application/json'
},
body: JSON.stringify({ prompt })
});
const data = await response.json();
const imageUrl = data.image_url; 
const imageContainer = document.createElement('div');
const img = document.createElement('img');
img.src = imageUrl;
imageContainer.appendChild(img);
document.getElementById('image-results').appendChild(imageContainer);
});
