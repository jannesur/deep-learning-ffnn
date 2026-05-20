console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

//Generieren von N=100 zufälligen, gleich-verteilten x Werte aus dem Intervall [-2,+2]
//Berechnen der dazugehörigen y-Werte

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

console.log(dataset);

// Datensatz mit Plotly visualisieren
Plotly.newPlot("dataset-plot", [
    {
        x: dataset.map(point => point.x),
        y: dataset.map(point => point.y),
        mode: "markers",
        type: "scatter",
        name: "Datenpunkte"
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