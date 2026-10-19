// noprotect
// 擦屏幕：手擦过的地方变成像素格；全部变像素后再擦，擦过的地方变回清晰。
// 依赖：p5.js（1.x / 2.x 都可以）+ MediaPipe Hand Landmarker（运行时从 CDN 加载）

let BLOCK = 5;             // 最小像素点大小（屏幕像素），运行时按 [ ] 调整
const COLOR_SMOOTH = 0.35; // 亮度时间平滑（0~1，越小越稳但越拖影）
const GRID_ALPHA = 0;      // 网格线透明度（0 = 不画网格）

// ---- 抖动像素风 ----
// 每套色系有两条从深到浅的颜色梯度：warm（暖）和 cool（冷），最后一个是背景色（两条要一样）。
// 原本偏暖的地方（皮肤、嘴唇）多用暖色墨点，偏冷/偏灰的地方（头发、背景）多用冷色墨点，中间混合。
// 两条梯度颜色数量要相同。按 P 切换，可以随意改
const TONES = [
  { name: 'Neon',
    warm: ['#2B1055', '#7A0F5C', '#D7194F', '#FF4F5E', '#FF9A3C', '#FFD84A', '#F2EDE0'],
    cool: ['#071A4A', '#123C9C', '#1F6FE0', '#00A3A3', '#4CD3B0', '#B4F0D2', '#F2EDE0'] },
  { name: 'Retro',
    warm: ['#2E1A16', '#6E2A1E', '#B5452B', '#E07A3F', '#E9B44C', '#F1D9A0', '#EBE4D3'],
    cool: ['#15233A', '#23466E', '#2F6F8F', '#4A9A8C', '#86C0A0', '#CFE3C2', '#EBE4D3'] },
  { name: 'Candy',
    warm: ['#3A0C3D', '#8C1C6E', '#E0337A', '#F6705B', '#FFA94D', '#FFE07A', '#F7F3EE'],
    cool: ['#1A1440', '#4B2FB8', '#6B6BF0', '#36A3E0', '#5FD9C8', '#BFF2DA', '#F7F3EE'] },
  { name: 'Red',
    warm: ['#4A000C', '#8E000E', '#D4060A', '#EE3B2A', '#F27A68', '#F2B2A6', '#C8C3C3'],
    cool: ['#4A000C', '#8E000E', '#D4060A', '#EE3B2A', '#F27A68', '#F2B2A6', '#C8C3C3'] },
];
const WARM_GAIN = 4;       // 冷暖区分的强度：越大越按原色分冷暖，越小两种墨点越混在一起
const WARM_BIAS = 0;       // >0 整体偏暖色，<0 整体偏冷色
const PATCH = 0.06;        // 冷暖色块的大小：越小色块越大越成片，越大越碎
const BIG_BLOCKS = [       // 平坦区域合并成大方块：size = 几个点合成一块，prob = 出现概率
  { size: 6, prob: 0.25 },
  { size: 3, prob: 0.35 },
];
const BLACK_POINT = 25;    // 比这个暗的都用最深色
const WHITE_POINT = 190;   // 比这个亮的都变成背景色（调小 → 背景更干净、墨点更少）
const GAMMA = 0.85;        // <1 整体偏亮（墨点更稀），>1 偏暗（墨点更密）
const FLAT_VAR = 120;      // 亮度方差小于这个才算"平坦"，可以合并成大方块
const SPRINKLE = 0.012;    // 背景里随机散落的彩色方块比例
let toneIdx = 0;

// ---- 擦除 ----
const HAND_PAD = 0.12;     // 手的轮廓向外扩一点（相对手掌大小），0 = 刚好贴着关键点
const MAX_JUMP = 2.5;      // 一帧内手移动超过 手掌大小×这个数 就认为是识别跳了，不补间
const MIN_SCORE = 0.6;     // 手部识别置信度低于这个就忽略
const MOUSE_R = 30;        // 鼠标擦除半径
const DONE_RATIO = 0.97;   // 覆盖到 97% 就算"全部完成"，自动补齐剩下的
const COOLDOWN_MS = 1200;  // 切换方向后短暂停一下，避免手立刻把刚完成的又擦回去

const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let video, landmarker;
let hands = [];            // [{ id: 'Left'/'Right', lm: 21 个归一化关键点 }]
let lastVideoTime = -1;
let loadMsg = 'Loading hand tracking model…';

let cols, rows;
let mask;                  // Uint8Array, 1 = 像素化, 0 = 清晰
let smallCanvas, smallCtx; // 低分辨率画面 = 像素格
let pixLayer, pixCtx;      // 全尺寸像素层（像素格 + 网格线）
let gridCanvas;            // 预先画好的网格线
let avg;                   // Float32Array，每个点平滑后的亮度
let palettes;              // 当前色系的 [暖色梯度 RGB, 冷色梯度 RGB]
let thresh, thresh2;       // 每个点固定的随机抖动阈值（固定不变，画面不会闪）
let lum, idx, ramp;        // 每个点的亮度、颜色编号、用暖色(0)还是冷色(1)梯度
let warmAvg;               // 每个点平滑后的冷暖值
let lumLo = 0, lumHi = 255; // 自动明暗拉伸的范围（平滑过的）

let target = 1;            // 当前擦的效果：1 = 擦成像素, 0 = 擦回清晰
let doneCount = 0;         // 已经是 target 状态的格子数
let cooldownUntil = 0;
let flashText = '';

let prevPts = {};          // 每只手上一帧的屏幕关键点，用于补间
let prevMouse = null;
let showHand = true;

async function loadLandmarker() {
  try {
    const { HandLandmarker, FilesetResolver } = await import(`${MP_URL}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    loadMsg = '';
  } catch (err) {
    console.error(err);
    loadMsg = 'Hand tracking failed to load (see console) · drag with the mouse to test';
  }
}

function detectHands() {
  const v = video.elt;
  if (!landmarker || v.readyState < 2 || v.currentTime === lastVideoTime) return;
  lastVideoTime = v.currentTime;
  const res = landmarker.detectForVideo(v, performance.now());
  hands = [];
  res.landmarks.forEach((lm, i) => {
    const h = res.handedness && res.handedness[i] && res.handedness[i][0];
    if (h && h.score < MIN_SCORE) return;
    // 按左右手区分，避免两只手顺序交换时在它们之间连线乱擦
    let id = h ? h.categoryName : 'hand' + i;
    if (hands.some((o) => o.id === id)) id += i;
    hands.push({ id, lm });
  });
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  video = createCapture(VIDEO);
  video.hide();
  loadLandmarker();

  smallCanvas = document.createElement('canvas');
  smallCtx = smallCanvas.getContext('2d');
  pixLayer = document.createElement('canvas');
  pixCtx = pixLayer.getContext('2d');
  gridCanvas = document.createElement('canvas');
  buildPalette();
  initGrid();

  textFont('sans-serif');
}

function initGrid() {
  cols = ceil(width / BLOCK);
  rows = ceil(height / BLOCK);
  mask = new Uint8Array(cols * rows);
  avg = null;
  lum = new Float32Array(cols * rows);
  idx = new Uint8Array(cols * rows);
  ramp = new Uint8Array(cols * rows);
  warmAvg = new Float32Array(cols * rows);
  thresh = new Float32Array(cols * rows);
  thresh2 = new Float32Array(cols * rows);
  for (let i = 0; i < thresh.length; i++) {
    thresh[i] = Math.random();
    // 冷暖用平滑噪声分区，颜色会成片出现，再加一点随机让边缘有颗粒感
    const nz = noise((i % cols) * PATCH, floor(i / cols) * PATCH);
    thresh2[i] = constrain(map(nz, 0.3, 0.7, 0, 1) * 0.75 + Math.random() * 0.25, 0, 1);
  }
  smallCanvas.width = cols;
  smallCanvas.height = rows;
  const pd = pixelDensity();
  pixLayer.width = gridCanvas.width = round(width * pd);
  pixLayer.height = gridCanvas.height = round(height * pd);
  buildGrid(pd);
  target = 1;
  doneCount = 0;
}

function buildGrid(pd) {
  const g = gridCanvas.getContext('2d');
  g.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
  if (GRID_ALPHA <= 0) return;
  g.fillStyle = `rgba(255,255,255,${GRID_ALPHA})`;
  const lw = max(1, round(pd));
  for (let c = 0; c <= cols; c++) {
    g.fillRect(round(c * BLOCK * pd), 0, lw, gridCanvas.height);
  }
  for (let r = 0; r <= rows; r++) {
    g.fillRect(0, round(r * BLOCK * pd), gridCanvas.width, lw);
  }
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildPalette() {
  palettes = [TONES[toneIdx].warm.map(hexToRgb), TONES[toneIdx].cool.map(hexToRgb)];
}

// 固定的伪随机数（同一个位置每帧都一样）
function hash(a, b) {
  const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  initGrid();
}

// 视频按 cover 方式铺满屏幕时的位置
function coverRect() {
  const vw = video.elt.videoWidth || 640;
  const vh = video.elt.videoHeight || 480;
  const s = max(width / vw, height / vh);
  const w = vw * s, h = vh * s;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

// 归一化关键点 → 屏幕坐标（镜像）
function toScreen(kp, r) {
  return { x: r.x + (1 - kp.x) * r.w, y: r.y + kp.y * r.h };
}

function draw() {
  background(0);
  const r = coverRect();
  const ctx = drawingContext;

  // 1. 清晰画面（镜像）
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video.elt, r.x, r.y, r.w, r.h);
  ctx.restore();

  // 2. 处理手的擦除：只擦手实际盖住的区域（关键点围成的轮廓）
  detectHands();
  const now = millis();
  const active = now >= cooldownUntil;
  const outlines = [];
  const seen = {};
  hands.forEach((h) => {
    const pts = h.lm.map((kp) => toScreen(kp, r));
    const size = dist(pts[0].x, pts[0].y, pts[9].x, pts[9].y); // 手腕到中指根
    const hull = expandHull(convexHull(pts), size * HAND_PAD);
    outlines.push({ hull, pts });
    seen[h.id] = true;
    if (active) {
      const prev = prevPts[h.id];
      // 和上一帧的轮廓合在一起擦，快速挥手也不会断；跳得太远说明识别跳了，只擦当前
      if (prev && dist(prev[0].x, prev[0].y, pts[0].x, pts[0].y) < size * MAX_JUMP) {
        wipePolygon(expandHull(convexHull(pts.concat(prev)), size * HAND_PAD));
      } else {
        wipePolygon(hull);
      }
    }
    prevPts[h.id] = pts;
  });
  for (const id in prevPts) {
    if (!seen[id]) delete prevPts[id];
  }

  if (mouseIsPressed) {
    if (active) {
      if (prevMouse) wipeLine(prevMouse.x, prevMouse.y, mouseX, mouseY, MOUSE_R);
      else wipeCircle(mouseX, mouseY, MOUSE_R);
    }
    prevMouse = { x: mouseX, y: mouseY };
  } else {
    prevMouse = null;
  }

  checkDone(now);

  // 3. 像素画面：缩小到 cols×rows → 像素风调色 → 无平滑放大 → 叠网格线
  renderPixels();
  if (pixLayer.width > 0) ctx.drawImage(pixLayer, 0, 0, width, height);

  // 4. UI
  if (showHand) {
    outlines.forEach(({ hull, pts }) => {
      noFill();
      stroke(255, 170);
      strokeWeight(1.5);
      beginShape();
      hull.forEach((p) => vertex(p.x, p.y));
      endShape(CLOSE);
      noStroke();
      fill(255, 200);
      pts.forEach((p) => circle(p.x, p.y, 4));
    });
    if (mouseIsPressed) {
      noFill();
      stroke(255, 170);
      circle(mouseX, mouseY, MOUSE_R * 2);
    }
  }
  drawUI(now);
}

// ---- 几何 ----
function convexHull(points) {
  const p = points.map((q) => ({ x: q.x, y: q.y })).sort((a, b) => a.x - b.x || a.y - b.y);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) {
      lower.pop();
    }
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) {
      upper.pop();
    }
    upper.push(q);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function expandHull(hull, pad) {
  let cx = 0, cy = 0;
  hull.forEach((p) => { cx += p.x; cy += p.y; });
  cx /= hull.length; cy /= hull.length;
  return hull.map((p) => {
    const d = dist(cx, cy, p.x, p.y) || 1;
    return { x: p.x + ((p.x - cx) / d) * pad, y: p.y + ((p.y - cy) / d) * pad };
  });
}

// 凸多边形内的格子（按格子中心判断）
function wipePolygon(poly) {
  if (poly.length < 3) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  poly.forEach((p) => {
    minX = min(minX, p.x); maxX = max(maxX, p.x);
    minY = min(minY, p.y); maxY = max(maxY, p.y);
  });
  const c0 = max(0, floor(minX / BLOCK)), c1 = min(cols - 1, floor(maxX / BLOCK));
  const r0 = max(0, floor(minY / BLOCK)), r1 = min(rows - 1, floor(maxY / BLOCK));
  const n = poly.length;
  for (let ry = r0; ry <= r1; ry++) {
    const y = (ry + 0.5) * BLOCK;
    for (let cx = c0; cx <= c1; cx++) {
      const x = (cx + 0.5) * BLOCK;
      let inside = true;
      for (let k = 0; k < n; k++) {
        const a = poly[k], b = poly[(k + 1) % n];
        if ((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x) < 0) { inside = false; break; }
      }
      if (inside) setCell(ry * cols + cx);
    }
  }
}

function wipeLine(x0, y0, x1, y1, rad) {
  const d = dist(x0, y0, x1, y1);
  const steps = max(1, ceil(d / (rad * 0.3)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    wipeCircle(lerp(x0, x1, t), lerp(y0, y1, t), rad);
  }
}

function wipeCircle(x, y, rad) {
  const c0 = max(0, floor((x - rad) / BLOCK));
  const c1 = min(cols - 1, floor((x + rad) / BLOCK));
  const r0 = max(0, floor((y - rad) / BLOCK));
  const r1 = min(rows - 1, floor((y + rad) / BLOCK));
  const rr = rad * rad;
  for (let ry = r0; ry <= r1; ry++) {
    const dy = (ry + 0.5) * BLOCK - y;
    for (let cx = c0; cx <= c1; cx++) {
      const dx = (cx + 0.5) * BLOCK - x;
      if (dx * dx + dy * dy <= rr) setCell(ry * cols + cx);
    }
  }
}

function setCell(i) {
  if (mask[i] !== target) {
    mask[i] = target;
    doneCount++;
  }
}

function checkDone(now) {
  if (doneCount / mask.length < DONE_RATIO) return;
  mask.fill(target);
  flashText = target === 1 ? 'All pixelated · keep wiping to reveal' : 'All clear · keep wiping to pixelate';
  target = 1 - target;
  doneCount = 0;
  cooldownUntil = now + COOLDOWN_MS;
}

// ---- 像素渲染 ----
function renderPixels() {
  // 直接从主画布（此时上面只有清晰的镜像画面）缩小采样，
  // 不从 video 读：部分浏览器把摄像头画面画进离屏 canvas 再读像素会得到全黑
  const src = drawingContext.canvas;
  smallCtx.globalCompositeOperation = 'copy';
  smallCtx.imageSmoothingEnabled = true;
  smallCtx.imageSmoothingQuality = 'high';
  smallCtx.drawImage(src, 0, 0, src.width, src.height, 0, 0, width / BLOCK, height / BLOCK);
  smallCtx.globalCompositeOperation = 'source-over';

  const img = smallCtx.getImageData(0, 0, cols, rows);
  const d = img.data;
  const n = cols * rows;
  updateLevels(d, n);
  const lvScale = 255 / max(40, lumHi - lumLo);

  const first = !avg;
  if (first) avg = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    // 自动明暗拉伸，画面偏暗/发灰时五官也能看清
    const y = (0.299 * d[o] + 0.587 * d[o + 1] + 0.114 * d[o + 2] - lumLo) * lvScale;
    if (first || !mask[i]) avg[i] = y; // 没被擦的点直接跟上当前亮度
    else avg[i] += (y - avg[i]) * COLOR_SMOOTH;
    // 冷暖：红减蓝，越正越暖
    const w = (d[o] - d[o + 2]) / 255;
    if (first || !mask[i]) warmAvg[i] = w;
    else warmAvg[i] += (w - warmAvg[i]) * COLOR_SMOOTH;
    lum[i] = pow(constrain((avg[i] - BLACK_POINT) / (WHITE_POINT - BLACK_POINT), 0, 1), GAMMA) * 255;
  }

  // 抖动：亮度落在两个颜色之间时，按固定随机阈值决定用深的还是浅的
  const last = palettes[0].length - 1;
  for (let i = 0; i < n; i++) {
    const t = (lum[i] / 255) * last;
    const k = min(floor(t), last - 1);
    idx[i] = t - k > thresh[i] ? k + 1 : k;
    // 冷暖也用抖动混合：越暖越可能用暖色墨点
    ramp[i] = 0.5 + (warmAvg[i] + WARM_BIAS) * WARM_GAIN > thresh2[i] ? 0 : 1;
  }

  // 平坦区域合并成大方块 + 背景里散落的方块
  BIG_BLOCKS.forEach(({ size, prob }) => {
    for (let by = 0; by + size <= rows; by += size) {
      for (let bx = 0; bx + size <= cols; bx += size) {
        const h = hash(bx * 7 + size, by * 13 + size);
        const sprinkle = h < SPRINKLE;
        if (!sprinkle && h > prob) continue;
        let sum = 0, sum2 = 0;
        for (let yy = 0; yy < size; yy++) {
          for (let xx = 0; xx < size; xx++) {
            const v = lum[(by + yy) * cols + bx + xx];
            sum += v; sum2 += v * v;
          }
        }
        const m = sum / (size * size);
        const isFlat = sum2 / (size * size) - m * m < FLAT_VAR;
        let k, rp;
        if (sprinkle && m > 200) {
          // 背景上的彩色方块，随机用暖色或冷色
          k = floor(hash(bx, by + 99) * (last - 1));
          rp = hash(bx + 5, by) < 0.5 ? 0 : 1;
        } else if (isFlat && !sprinkle) {
          k = round((m / 255) * last);
          rp = ramp[by * cols + bx];
        } else continue;
        for (let yy = 0; yy < size; yy++) {
          for (let xx = 0; xx < size; xx++) {
            idx[(by + yy) * cols + bx + xx] = k;
            ramp[(by + yy) * cols + bx + xx] = rp;
          }
        }
      }
    }
  });

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (!mask[i]) { d[o + 3] = 0; continue; }
    const c = palettes[ramp[i]][idx[i]];
    d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
  }
  smallCtx.putImageData(img, 0, 0);

  const pd = pixLayer.width / width;
  pixCtx.globalCompositeOperation = 'copy';
  pixCtx.imageSmoothingEnabled = false;
  pixCtx.drawImage(smallCanvas, 0, 0, cols * BLOCK * pd, rows * BLOCK * pd);
  if (GRID_ALPHA > 0) {
    pixCtx.globalCompositeOperation = 'source-atop';
    pixCtx.drawImage(gridCanvas, 0, 0);
  }
  pixCtx.globalCompositeOperation = 'source-over';
}

// 用亮度直方图找 2% / 98% 的亮度，作为拉伸范围
function updateLevels(d, n) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    hist[(0.299 * d[o] + 0.587 * d[o + 1] + 0.114 * d[o + 2]) | 0]++;
  }
  let lo = 0, hi = 255, acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= n * 0.02) { lo = v; break; }
  }
  acc = 0;
  for (let v = 255; v >= 0; v--) {
    acc += hist[v];
    if (acc >= n * 0.02) { hi = v; break; }
  }
  lumLo += (lo - lumLo) * 0.1;
  lumHi += (hi - lumHi) * 0.1;
}

function drawUI(now) {
  const ratio = doneCount / mask.length;
  const barW = min(320, width - 32);
  const bx = (width - barW) / 2, by = height - 36;
  noStroke();
  fill(0, 120);
  rect(bx - 8, by - 26, barW + 16, 40, 8);
  fill(255, 60);
  rect(bx, by, barW, 6, 3);
  fill(255);
  rect(bx, by, barW * min(1, ratio / DONE_RATIO), 6, 3);
  textSize(13);
  textAlign(CENTER, BOTTOM);
  text((target === 1 ? 'Wave to wipe → Pixelate' : 'Wave to wipe → Reveal') + ' · ' + TONES[toneIdx].name, width / 2, by - 6);

  if (now < cooldownUntil + 800 && flashText) {
    const a = map(now, cooldownUntil, cooldownUntil + 800, 255, 0, true);
    fill(0, a * 0.5);
    rect(0, height / 2 - 40, width, 80);
    fill(255, a);
    textSize(28);
    textAlign(CENTER, CENTER);
    text(flashText, width / 2, height / 2);
  }

  if (loadMsg || (hands.length === 0 && !mouseIsPressed)) {
    fill(255, 180);
    textSize(13);
    textAlign(CENTER, TOP);
    text(loadMsg || 'Raise your hand to the camera · R reset · H hand outline · P palette · [ ] pixel size · or drag with the mouse', width / 2, 16);
  }
}

function keyPressed() {
  if (key === 'r' || key === 'R') initGrid();
  if (key === 'h' || key === 'H') showHand = !showHand;
  if (key === 'p' || key === 'P') {
    toneIdx = (toneIdx + 1) % TONES.length;
    buildPalette();
  }
  if (key === '[' || key === ']') {
    BLOCK = constrain(BLOCK + (key === ']' ? 1 : -1), 2, 40);
    initGrid();
  }
}
