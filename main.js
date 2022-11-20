let model;
let canvas;
const classNames = [];
const coords = [];
let mousePressed = false;
const loc = 'model2/class_names.txt'
const modelJSON = 'model2/model.json'


function recordCoor(event) {
    const pointer = canvas.getPointer(event.e);
    const posX = pointer.x;
    const posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousePressed) {
        coords.push(pointer)
    }
}

/*
get the best bounding box by trimming around the drawing
*/
function getMinBox() {
    const coorX = coords.map(function (p) {
        return p.x
    });
    const coorY = coords.map(function (p) {
        return p.y
    });

    const min = {
        x: Math.min.apply(null, coorX),
        y: Math.min.apply(null, coorY)
    }
    const max = {
        x: Math.max.apply(null, coorX),
        y: Math.max.apply(null, coorY)
    }

    return {
        min,
        max
    }
}


function getImageData() {
    const mbb = getMinBox()
    const dpi = window.devicePixelRatio
    const imgData = canvas
        .contextContainer
        .getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
            (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
    return imgData
}

/*
get the prediction 
*/
function getFrame(callback) {
    //make sure we have at least two recorded coordinates 
    if (coords.length >= 2) {
        const imgData = getImageData()
        const pred = model.predict(preprocess(imgData)).dataSync()
        const indices = findIndicesOfMax(pred, 5)
        const probs = findTopValues(pred, indices)
        const names = getClassNames(indices)
        const data = names.map((name,i)=>{
            return {
                name,
                prob:probs[i]
            }
        })
        callback(data)
    }

}

function getClassNames(indices) {
    return indices.reduce((result, key, i) => {
        result[i] = classNames[key]
        return result
    }, [])
}

async function loadDict() {
    return fetch(loc).then(response => response.text()).then(data => success(data));
}


function success(data) {
    data.split(/\n/).forEach((item, i) => classNames[i] = item)
}

function findIndicesOfMax(inp, count) {
    const outp = [];
    for (let i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function (a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}

function findTopValues(inp, indices) {
    return indices.map(key => inp[key])
}

function preprocess(imgData) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imgData, 1)
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()
        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));
        const batched = normalized.expandDims(0)
        return batched
    })
}

async function start(callback) {
    init(callback)
    if (!model)
        model = await tf.loadLayersModel(modelJSON)
    model.predict(tf.zeros([1, 28, 28, 1]))

    //allow drawing on the canvas 
    canvas.isDrawingMode = 1;

    //load the class names
    await loadDict()
    return {
        clear
    }
}



/*
clear the canvs 
*/
function clear() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords.length = 0;
}



function init(callback) {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 10;
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function (e) {
        getFrame(callback);
        mousePressed = false
    });
    canvas.on('mouse:down', function (e) {
        mousePressed = true
    });
    canvas.on('mouse:move', recordCoor);

}
