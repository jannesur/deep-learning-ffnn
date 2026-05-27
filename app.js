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


// Modell ohne Rauschen erstellen

const cleanModel = tf.sequential();

cleanModel.add(tf.layers.dense({

    units: 100,

    activation: "relu",

    inputShape: [1]

}));

cleanModel.add(tf.layers.dense({

    units: 100,

    activation: "relu"

}));

cleanModel.add(tf.layers.dense({

    units: 1,

    activation: "linear"

}));

cleanModel.compile({

    optimizer: tf.train.adam(0.01),

    loss: "meanSquaredError"

});


// Feed-Forward Neural Network erstellen

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


// Best-Fit Modell erstellen

const bestFitModel = tf.sequential();

bestFitModel.add(tf.layers.dense({

    units: 100,

    activation: "relu",

    inputShape: [1]

}));

bestFitModel.add(tf.layers.dense({

    units: 100,

    activation: "relu"

}));

bestFitModel.add(tf.layers.dense({

    units: 1,

    activation: "linear"

}));

bestFitModel.compile({

    optimizer: tf.train.adam(0.01),

    loss: "meanSquaredError"

});


// Overfit Modell erstellen

const overfitModel = tf.sequential();

overfitModel.add(tf.layers.dense({

    units: 100,

    activation: "relu",

    inputShape: [1]

}));

overfitModel.add(tf.layers.dense({

    units: 100,

    activation: "relu"

}));

overfitModel.add(tf.layers.dense({

    units: 1,

    activation: "linear"

}));

overfitModel.compile({

    optimizer: tf.train.adam(0.01),

    loss: "meanSquaredError"

});


// Modelle trainieren

async function trainModel() {

    // Modell ohne Rauschen trainieren

    await cleanModel.fit(

        cleanTrainingInputs,
        cleanTrainingLabels,

        {
            epochs: 100,
            batchSize: 32,
            shuffle: true
        }

    );


    // Clean Modell visualisieren

    const cleanTrainPredictions =
        cleanModel.predict(cleanTrainingInputs);

    const cleanTestPredictions =
        cleanModel.predict(cleanTestInputs);

    const cleanTrainPredictionValues =
        await cleanTrainPredictions.array();

    const cleanTestPredictionValues =
        await cleanTestPredictions.array();

    Plotly.newPlot("clean-train-prediction-plot", [

        {
            x: trainingData.map(point => point.x),

            y: trainingData.map(point => point.y),

            mode: "markers",

            type: "scatter",

            name: "Trainingsdaten"
        },

        {
            x: trainingData.map(point => point.x),

            y: cleanTrainPredictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Vorhersage"
        }

    ], {

        title: "Clean Modell - Trainingsdaten",

        xaxis: {
            title: "x"
        },

        yaxis: {
            title: "y"
        }

    });

    Plotly.newPlot("clean-test-prediction-plot", [

        {
            x: testData.map(point => point.x),

            y: testData.map(point => point.y),

            mode: "markers",

            type: "scatter",

            name: "Testdaten"
        },

        {
            x: testData.map(point => point.x),

            y: cleanTestPredictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Vorhersage"
        }

    ], {

        title: "Clean Modell - Testdaten",

        xaxis: {
            title: "x"
        },

        yaxis: {
            title: "y"
        }

    });


    // Clean Loss berechnen

    const cleanTrainLossTensor = cleanModel.evaluate(
        cleanTrainingInputs,
        cleanTrainingLabels
    );

    const cleanTestLossTensor = cleanModel.evaluate(
        cleanTestInputs,
        cleanTestLabels
    );

    const cleanTrainLoss =
        cleanTrainLossTensor.dataSync()[0];

    const cleanTestLoss =
        cleanTestLossTensor.dataSync()[0];

    document.getElementById("clean-train-loss").textContent =
        cleanTrainLoss.toFixed(6);

    document.getElementById("clean-test-loss").textContent =
        cleanTestLoss.toFixed(6);


    // Normales Modell trainieren

    await model.fit(

        trainingInputs,
        trainingLabels,

        {
            epochs: 100,
            batchSize: 32,
            shuffle: true
        }

    );


    // Best-Fit Modell trainieren

    await bestFitModel.fit(

        trainingInputs,
        trainingLabels,

        {
            epochs: 50,
            batchSize: 32,
            shuffle: true
        }

    );


    // Overfit Modell trainieren

    await overfitModel.fit(

        trainingInputs,
        trainingLabels,

        {
            epochs: 150,
            batchSize: 32,
            shuffle: true
        }

    );


    // Modellvorhersagen visualisieren

    const modelPredictions =
        model.predict(trainingInputs);

    const bestFitPredictions =
        bestFitModel.predict(trainingInputs);

    const overfitPredictions =
        overfitModel.predict(trainingInputs);

    const modelPredictionValues =
        await modelPredictions.array();

    const bestFitPredictionValues =
        await bestFitPredictions.array();

    const overfitPredictionValues =
        await overfitPredictions.array();

    Plotly.newPlot("prediction-plot", [

        {
            x: noisyTrainingData.map(point => point.x),

            y: noisyTrainingData.map(point => point.y),

            mode: "markers",

            type: "scatter",

            name: "Trainingsdaten"
        },

        {
            x: noisyTrainingData.map(point => point.x),

            y: modelPredictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Normales Modell"
        },

        {
            x: noisyTrainingData.map(point => point.x),

            y: bestFitPredictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Best-Fit Modell"
        },

        {
            x: noisyTrainingData.map(point => point.x),

            y: overfitPredictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Overfit Modell"
        }

    ], {

        title: "Vergleich der Modellvorhersagen",

        xaxis: {
            title: "x"
        },

        yaxis: {
            title: "y"
        }

    });


    // Trainings- und Test-Loss berechnen

    const trainLossTensor =
        model.evaluate(trainingInputs, trainingLabels);

    const testLossTensor =
        model.evaluate(testInputs, testLabels);

    const trainLoss =
        trainLossTensor.dataSync()[0];

    const testLoss =
        testLossTensor.dataSync()[0];

    document.getElementById("train-loss").textContent =
        trainLoss.toFixed(6);

    document.getElementById("test-loss").textContent =
        testLoss.toFixed(6);

}

trainModel();


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