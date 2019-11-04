const debug = require('debug')('digit-classifier')
const tf = require('@tensorflow/tfjs')

const config = require('../config')

module.exports = class DigitClassifier {

    constructor () {
        /**
         * Sequential model
         * @description Each layer is layer is only connected to the next layer
         */
        this.model = tf.sequential()

        this.model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],

            activation: 'relu',
            filters: 32,
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            kernelSize: [5, 5],
            padding: 'same',
        }))

        this.model.add(tf.layers.conv2d({
            activation: 'relu',
            filters: 32,
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            kernelSize: [5, 5],
            padding: 'same',
        }))

        this.model.add(tf.layers.maxPool2d({
        }))

        this.model.add(tf.layers.dropout({
            rate: .25,
        }))

        this.model.add(tf.layers.conv2d({
            activation: 'relu',
            filters: 64,
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            kernelSize: [3, 3],
            padding: 'same',
        }))

        this.model.add(tf.layers.conv2d({
            activation: 'relu',
            filters: 64,
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            kernelSize: [3, 3],
            padding: 'same',
        }))

        this.model.add(tf.layers.maxPool2d({
        }))

        this.model.add(tf.layers.dropout({
            rate: .25,
        }))

        this.model.add(tf.layers.flatten({
            dataFormat: 'channelsLast',
        }))

        this.model.add(tf.layers.dense({
            activation: 'relu',
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            units: 256,
        }))

        this.model.add(tf.layers.dropout({
            rate: .5,
        }))

        this.model.add(tf.layers.dense({
            activation: 'softmax',
            kernelInitializer: tf.initializers.varianceScaling({
                distribution: 'uniform',
                mode: 'fanAvg',
            }),
            units: 10,
        }))

        this.model.compile({
            optimizer: tf.train.rmsprop(.0000625),
            loss: tf.metrics.categoricalCrossentropy,
            metrics: ['accuracy'],
        })
        
        this.model.summary()
    }

    /**
     * 
     * @param {string} handler - tensorflow IO handler eg.: file:// or downloads://
     */
    async loadModel (handler) {
        this.model = await tf.loadModel(handler)

        this.model.compile()
    }

    /**
     * 
     * @param {string} handler - tensorflow IO handler eg.: file:// or downloads://
     */
    async saveModel (handler) {
        debug('Saving model to ' + handler)
        await this.model.save(handler)
        debug('Model saving finished')
    }

    /**
     * 
     * @param {number[]} images - 1d array with size of labels.length*img_height*img_height*img_depth
     * @param {number[]} labels - 1d array of correct values, with length of batch size
     * @param {number[]} testImages - 1d array with size of labels.length*img_height*img_height*img_depth
     * @param {number[]} testLabels - 1d array of correct values, with length of batch size
     */
    async train (images, labels, testImages, testLabels) {
        debug('Creating training images tensor')
        const xs = tf.tensor4d(images, [
            config.batchSize,
            config.imageHeight,
            config.imageWidth,
            config.imageDepth
        ])
        debug('Creating training labels onehot tensor')
        const ys = tf.oneHot(labels, 10) // 3, 5 => [0, 0, 0, /* 3rd=1 */ 1, 0, 0]

        let validationData
        if (testImages && testLabels) {
            debug('Creating testing tensors')
            validationData = [
                tf.tensor4d(testImages, [
                    config.testingBatchSize,
                    config.imageHeight,
                    config.imageWidth,
                    config.imageDepth
                ]),tf.oneHot(testLabels, 10)
              ];
        }

        debug('Starting model train')
        await this.model.fit(xs, ys, {
            epochs: 1,
            validationData,
        })
        debug('Training done, cleanup')
        /* clean up memory */
        // TODO: use tf.tidy(async function)
        xs.dispose()
        ys.dispose()
        debug('done')
    }

    // predict () {

    // }
}