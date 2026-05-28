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


// Gaussian Noise erzeugen

function generateGaussianNoise(mean = 0, variance = 0.05) {

    const standardDeviation = Math.sqrt(variance);

    const randomValue1 = Math.random();

    const randomValue2 = Math.random();

    const randomNormalValue =
        Math.sqrt(-2 * Math.log(randomValue1)) *
        Math.cos(2 * Math.PI * randomValue2);

    return mean + standardDeviation * randomNormalValue;
}


// Trainingsdaten und Testdaten künstlich verrauschen

const noisyTrainingData = trainingData.map(point => {

    return {
        x: point.x,
        y: point.y + generateGaussianNoise()
    };
});

const noisyTestData = testData.map(point => {

    return {
        x: point.x,
        y: point.y + generateGaussianNoise()
    };
});


// Trainings- und Testdaten vorbereiten

const cleanTrainingInputs = tf.tensor2d(
    trainingData.map(point => [point.x])
);

const cleanTrainingLabels = tf.tensor2d(
    trainingData.map(point => [point.y])
);

const cleanTestInputs = tf.tensor2d(
    testData.map(point => [point.x])
);

const cleanTestLabels = tf.tensor2d(
    testData.map(point => [point.y])
);

const trainingInputs = tf.tensor2d(
    noisyTrainingData.map(point => [point.x])
);

const trainingLabels = tf.tensor2d(
    noisyTrainingData.map(point => [point.y])
);

const testInputs = tf.tensor2d(
    noisyTestData.map(point => [point.x])
);

const testLabels = tf.tensor2d(
    noisyTestData.map(point => [point.y])
);


// Modell erstellen

function createModel() {

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 100,
        activation: "relu",
        inputShape: [1]
    }));

    model.add(tf.layers.dense({
        units: 100,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: "linear"
    }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "meanSquaredError"
    });

    return model;
}


const cleanModel = createModel();

const bestFitModel = createModel();

const overfitModel = createModel();


// Hilfsfunktion: x-Werte für eine glatte Vorhersage-Kurve erzeugen

function createCurveXValues() {

    const curveXValues = [];

    for (let i = 0; i <= 200; i++) {

        const x = -2 + (4 * i / 200);

        curveXValues.push(x);
    }

    return curveXValues;
}


// Plot-Funktion für Modellvorhersagen

async function plotPredictions(plotId, originalData, model, title, dataName) {

    const curveXValues = createCurveXValues();

    const curveInputs = tf.tensor2d(
        curveXValues.map(x => [x])
    );

    const predictions = model.predict(curveInputs);

    const predictionValues = await predictions.array();

    Plotly.newPlot(plotId, [

        {
            x: originalData.map(point => point.x),
            y: originalData.map(point => point.y),
            mode: "markers",
            type: "scatter",
            name: dataName
        },

        {
            x: curveXValues,
            y: predictionValues.map(value => value[0]),
            mode: "lines",
            type: "scatter",
            name: "Vorhersage-Kurve"
        }

    ], {

        title: title,

        xaxis: {
            title: "x"
        },

        yaxis: {
            title: "y"
        }

    });

    curveInputs.dispose();
    predictions.dispose();
}


// Loss berechnen

function calculateLoss(model, inputs, labels) {

    const lossTensor = model.evaluate(inputs, labels);

    const lossValue = lossTensor.dataSync()[0];

    lossTensor.dispose();

    return lossValue;
}


// Modell trainieren und Loss-Verlauf speichern

async function trainModelAndSaveLossHistory(model, inputs, labels, epochs) {

    const lossHistory = [];

    await model.fit(

        inputs,
        labels,

        {
            epochs: epochs,
            batchSize: 32,
            shuffle: true,

            callbacks: {
                onEpochEnd: function(epoch, logs) {

                    lossHistory.push(logs.loss);
                }
            }
        }

    );

    return lossHistory;
}


// Loss-Verlauf plotten

function plotLossHistory(plotId, lossHistory, title) {

    const epochNumbers = [];

    for (let i = 0; i < lossHistory.length; i++) {

        epochNumbers.push(i + 1);
    }

    Plotly.newPlot(plotId, [

        {
            x: epochNumbers,
            y: lossHistory,
            mode: "lines",
            type: "scatter",
            name: "Trainings-Loss"
        }

    ], {

        title: title,

        xaxis: {
            title: "Epoch"
        },

        yaxis: {
            title: "MSE"
        }

    });
}


// Kleine Hilfsfunktion: Loss-Plot-Div erstellen, falls es noch nicht existiert

function createLossPlotDivIfNeeded(parentId, plotId) {

    const parentElement = document.getElementById(parentId);

    if (document.getElementById(plotId) === null) {

        const lossPlotDiv = document.createElement("div");

        lossPlotDiv.id = plotId;

        lossPlotDiv.style.width = "100%";

        lossPlotDiv.style.height = "350px";

        lossPlotDiv.style.marginTop = "20px";

        parentElement.appendChild(lossPlotDiv);
    }
}


// Modelle trainieren

async function trainModels() {

    // Modell ohne Rauschen trainieren

    const cleanLossHistory = await trainModelAndSaveLossHistory(
        cleanModel,
        cleanTrainingInputs,
        cleanTrainingLabels,
        300
    );


    // Best-Fit Modell trainieren

    const bestFitLossHistory = await trainModelAndSaveLossHistory(
        bestFitModel,
        trainingInputs,
        trainingLabels,
        150
    );


    // Overfit Modell trainieren

    const overfitLossHistory = await trainModelAndSaveLossHistory(
        overfitModel,
        trainingInputs,
        trainingLabels,
        800
    );


    // R2 visualisieren

    await plotPredictions(
        "clean-train-prediction-plot",
        trainingData,
        cleanModel,
        "Clean Modell - Trainingsdaten",
        "Trainingsdaten"
    );

    await plotPredictions(
        "clean-test-prediction-plot",
        testData,
        cleanModel,
        "Clean Modell - Testdaten",
        "Testdaten"
    );


    // R3 visualisieren

    await plotPredictions(
        "bestfit-train-prediction-plot",
        noisyTrainingData,
        bestFitModel,
        "Best-Fit Modell - Trainingsdaten",
        "Trainingsdaten"
    );

    await plotPredictions(
        "bestfit-test-prediction-plot",
        noisyTestData,
        bestFitModel,
        "Best-Fit Modell - Testdaten",
        "Testdaten"
    );


    // R4 visualisieren

    await plotPredictions(
        "overfit-train-prediction-plot",
        noisyTrainingData,
        overfitModel,
        "Overfit Modell - Trainingsdaten",
        "Trainingsdaten"
    );

    await plotPredictions(
        "overfit-test-prediction-plot",
        noisyTestData,
        overfitModel,
        "Overfit Modell - Testdaten",
        "Testdaten"
    );


    // Loss-Werte anzeigen

    document.getElementById("clean-train-loss").textContent =
        calculateLoss(cleanModel, cleanTrainingInputs, cleanTrainingLabels).toFixed(6);

    document.getElementById("clean-test-loss").textContent =
        calculateLoss(cleanModel, cleanTestInputs, cleanTestLabels).toFixed(6);

    document.getElementById("bestfit-train-loss").textContent =
        calculateLoss(bestFitModel, trainingInputs, trainingLabels).toFixed(6);

    document.getElementById("bestfit-test-loss").textContent =
        calculateLoss(bestFitModel, testInputs, testLabels).toFixed(6);

    document.getElementById("overfit-train-loss").textContent =
        calculateLoss(overfitModel, trainingInputs, trainingLabels).toFixed(6);

    document.getElementById("overfit-test-loss").textContent =
        calculateLoss(overfitModel, testInputs, testLabels).toFixed(6);


    // Loss-Verlauf zusätzlich visualisieren

    createLossPlotDivIfNeeded("clean-loss-output", "clean-loss-plot");
    createLossPlotDivIfNeeded("bestfit-loss-output", "bestfit-loss-plot");
    createLossPlotDivIfNeeded("overfit-loss-output", "overfit-loss-plot");

    plotLossHistory(
        "clean-loss-plot",
        cleanLossHistory,
        "Loss-Verlauf Clean Modell"
    );

    plotLossHistory(
        "bestfit-loss-plot",
        bestFitLossHistory,
        "Loss-Verlauf Best-Fit Modell"
    );

    plotLossHistory(
        "overfit-loss-plot",
        overfitLossHistory,
        "Loss-Verlauf Overfit Modell"
    );


    // Rote Trainingshinweise ausblenden

    const trainingStatusTexts = document.querySelectorAll(".training-status");

    for (const trainingStatusText of trainingStatusTexts) {

        trainingStatusText.style.display = "none";
    }
}


// Unverrauschten Datensatz visualisieren

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


// Verrauschten Datensatz visualisieren

Plotly.newPlot("noisy-dataset-plot", [

    {
        x: noisyTrainingData.map(point => point.x),
        y: noisyTrainingData.map(point => point.y),
        mode: "markers",
        type: "scatter",
        name: "Trainingsdaten"
    },

    {
        x: noisyTestData.map(point => point.x),
        y: noisyTestData.map(point => point.y),
        mode: "markers",
        type: "scatter",
        name: "Testdaten"
    }

], {

    title: "Verrauschter Datensatz",

    xaxis: {
        title: "x"
    },

    yaxis: {
        title: "y"
    }

});


// Training starten

trainModels();