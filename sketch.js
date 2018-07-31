let points = [];
let model;
let chart;

/* exported setup */
/**
 * p5 setup function
 */
function setup() {
  model = new KerasJS.Model({
    filepath: 'models/conv2d.bin',
    // filepath: 'models/flat_nn.bin',
    gpu: true,
  });

  const ctx = document.getElementById('chart');
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      datasets: [{
        label: '%',
        data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      }],
    },
    options: {
      responsive: false,
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: true,
            suggestedMax: 100,
          },
        }],
      },
    },
  });

  createCanvas(400, 400).parent('canvas-holder');
  background(255);
  stroke(0);
  strokeWeight(1);
  noFill();
  rect(0, 0, width - 1, height - 1);

  select('#clearBtn').mousePressed(() => {
    background(255);
    stroke(0);
    strokeWeight(1);
    noFill();
    rect(0, 0, width - 1, height - 1);

    chart.data.datasets[0].data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    chart.update();
  });
}

/* exported draw */
/**
 * p5 draw function
 */
function draw() {
  if (mouseIsPressed && (
      mouseX >= 0 && mouseX <= width &&
      mouseY >= 0 && mouseY <= height)) {
    points.push(createVector(mouseX, mouseY));
  }

  stroke(0);
  strokeWeight(30);
  noFill();
  beginShape();
  for (let point of points) {
    vertex(point.x, point.y);
  }
  endShape();
}

/* exported mouseReleased */
/**
 * p5 mouseReleased function
 */
function mouseReleased() {
  if (mouseX > width || mouseX < 0 ||
      mouseY > height || mouseY < 0) {
    return;
  }

  points = [];

  // Clear the top left corner
  stroke(255);
  strokeWeight(2);
  fill(255);
  rect(5, 5, 28, 28);

  // Don't know a better way, so just copy the pixel array
  const img = createImage(width, height);
  loadPixels();
  img.loadPixels();
  for (let i = 0; i < pixels.length; i++) {
    img.pixels[i] = pixels[i];
  }
  img.updatePixels();

  // Resize to 28x28
  img.resize(28, 28);

  // Display scaled image
  stroke(0);
  strokeWeight(2);
  rect(5, 5, 28, 28);
  image(img, 5, 5);

  // Extract
  img.loadPixels();
  let x = [];
  let idx = 0;
  for (let i = 0; i < 784; i++) {
    x.push(img.pixels[idx] / 255);
    idx += 4;
  }
  x = new Float32Array(x);

  // Predict
  model.ready()
  .then(() => model.predict({
    input: x,
  }))
  .then((outputData) => {
    chart.data.datasets[0].data = outputData.output.map((x) => x * 100);
    chart.update();
  })
  .catch((err) => {
    console.error(err);
  });
}
