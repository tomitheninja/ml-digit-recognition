const tf = require('@tensorflow/tfjs')
const debug = require('debug')('App')



const config = require('./config')
const DigitClassifier = require('./model/digit-classifier')
const getMNISTData = require('./utils/getMNISTData')

module.exports = async function App () {

    debug('Creating digitClassifier')
    const classifier = new DigitClassifier()

    debug('Training network')
    for (let i = 0; i < config.numTrainingIterations; i++) {
        for (let j = 0; j < config.numTrainBatches; j++) {

            debug(`${i} / ${config.numTrainingIterations}  ------- ${j} / ${config.numTrainBatches}`)


            const trainingSize = config.batchSize
            const trainingStart = j * trainingSize
            const [images, labels] = getMNISTData('training', trainingStart, trainingSize)


            const shouldTest = true // Math.random() < .3
            let testImages, testLabels
            if (shouldTest) {
                const [testingStart, testingAmount] = config.randomTestingBatch()
                const [tmpImages, tmpLabels] = getMNISTData('testing', testingStart, testingAmount)
                testImages = tmpImages
                testLabels = tmpLabels
            }
            debug('Training network')
            await classifier.train(images, labels, testImages, testLabels)
            debug('Training network with batch done')

        }
    }

    debug('Saving model')
    classifier.saveModel(config.snapshotDir +  new Date().toISOString().slice(0, 19))

    debug('Done')

}