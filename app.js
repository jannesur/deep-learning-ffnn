console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);


// Generieren von N=100 zufälligen, gleich-verteilten x-Werten aus dem Intervall [-2,+2]
// Berechnen der dazugehörigen y-Werte

const DATASET_SIZE = 100;

const xValues = [];

function calculateY(x) {

    return (
        0.5 *
        (x + 0.8) *
        (x + 1.8) *
        (x - 0.2) *
        (x - 0.3) *
        (x - 1.9)
        + 1
    );
}

for (let i = 0; i < DATASET_SIZE; i++) {

    const x = Math.random() * 4 - 2;

    xValues.push(x);
}

const dataset = [];

for (const x of xValues) {

    const y = calculateY(x);

    dataset.push({
        x: x,
        y: y
    });
}


// Datensatz zufällig mischen und in Trainingsdaten und Testdaten aufteilen

dataset.sort(() => Math.random() - 0.5);

const trainingData = dataset.slice(0, DATASET_SIZE / 2);

const testData = dataset.slice(DATASET_SIZE / 2);

console.log("Training Data:", trainingData);
console.log("Test Data:", testData);


// Datensatz mit Plotly visualisieren

Plotly.newPlot("dataset-plot", [

    {
        x: trainingData.map(point => point.x),
        y: trainingData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Trainingsdaten"
    },

    {
        x: testData.map(point => point.x),
        y: testData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Testdaten"
    }

], {

    title: "Unverrauschter Datensatz",

    xaxis: {
        title: "x"
    },

    yaxis: {
        title: "y"
    }

});