// Object example from book. Page 129, 159.
// JitterBug -> two different creatures

var bug;
var jelly;

function setup() {
  createCanvas(480, 120);
  // Ladybug
  bug = new Ladybug(width/2, height/2, 30);
  // Jellyfish
  jelly = new Jellyfish(100, 60, 40);
}

function draw() {
  background(220); // wenn background bei setup, dann schlange
  bug.move();
  bug.display();
  jelly.move();
  jelly.display();
}

// Creature 1: a ladybug that jitters fast in every direction
function Ladybug(tempX, tempY, tempDiameter){
  this.x = tempX;
  this.y = tempY;
  this.diameter = tempDiameter;
  this.speed = 2.5;

  this.move = function(){
    this.x += random(-this.speed, this.speed);
    this.y += random(-this.speed, this.speed);
    // Constrain keeps the creature from leaving canvas
    this.x = constrain(this.x, 0, width);
    this.y = constrain(this.y, 0, height);
  };

  this.display = function(){
    var d = this.diameter;
    noStroke();
    // Head
    fill(0);
    ellipse(this.x, this.y - d * 0.45, d * 0.5, d * 0.5);
    // Red body
    fill(220, 30, 30);
    ellipse(this.x, this.y, d, d);
    // Line down the back
    stroke(0);
    line(this.x, this.y - d / 2, this.x, this.y + d / 2);
    // Spots
    noStroke();
    fill(0);
    ellipse(this.x - d * 0.2, this.y - d * 0.1, d * 0.2, d * 0.2);
    ellipse(this.x + d * 0.2, this.y - d * 0.1, d * 0.2, d * 0.2);
    ellipse(this.x - d * 0.15, this.y + d * 0.2, d * 0.15, d * 0.15);
    ellipse(this.x + d * 0.15, this.y + d * 0.2, d * 0.15, d * 0.15);
  };
}

// Creature 2: a jellyfish that drifts slowly and wiggles its tentacles
function Jellyfish(tempX, tempY, tempDiameter){
  this.x = tempX;
  this.y = tempY;
  this.diameter = tempDiameter;
  this.speed = 1;

  this.move = function(){
    this.x += random(-this.speed, this.speed);
    this.y += random(-this.speed, this.speed);
    // Constrain keeps the creature from leaving canvas
    this.x = constrain(this.x, 0, width);
    this.y = constrain(this.y, 0, height);
  };

  this.display = function(){
    var d = this.diameter;
    // Tentacles
    stroke(170, 90, 200);
    strokeWeight(2);
    for (var i = -2; i <= 2; i++) {
      var tx = this.x + i * d * 0.15;
      line(tx, this.y, tx + random(-3, 3), this.y + d * 0.7);
    }
    strokeWeight(1);
    // Dome-shaped body
    noStroke();
    fill(200, 140, 230);
    arc(this.x, this.y, d, d, PI, TWO_PI);
    // Eyes
    fill(0);
    ellipse(this.x - d * 0.15, this.y - d * 0.15, 4, 4);
    ellipse(this.x + d * 0.15, this.y - d * 0.15, 4, 4);
  };
}
