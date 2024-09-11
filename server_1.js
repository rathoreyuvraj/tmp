const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Set the view engine to EJS
app.set('view engine', 'ejs');

// Serve static files from the public folder
app.use(express.static('public'));

// Helper function to get image file paths from a directory
const getImagesFromDir = (dir) => {
  const directoryPath = path.join(__dirname, 'public', 'images', dir);
  return fs.readdirSync(directoryPath).map(file => `/images/${dir}/${file}`);
};

// Route to render the HTML page with images
app.get('/', (req, res) => {
  // Get images from four directories
  const topImage = getImagesFromDir('dir1')[0]; // Select one image from the top directory
  const bottomImages = [
    getImagesFromDir('dir2')[0],
    getImagesFromDir('dir3')[0],
    getImagesFromDir('dir4')[0],
    getImagesFromDir('dir4')[1], // Assuming two images in dir4
  ];

  // Render the HTML with the images
  res.render('index', { topImage, bottomImages });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
