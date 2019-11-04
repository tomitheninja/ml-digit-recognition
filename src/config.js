module.exports = {
    /* library defined */
    maxTrainingSetSize: 65000,
    maxTestingSetSize: 10000,
    /* image shape */
    imageHeight: 28,
    imageWidth: 28,
    imageDepth: 1,

    /* training */
    trainingsetSize: 65000,
    numTrainBatches: 10,
    get batchSize () { return Math.floor(this.trainingsetSize / this.numTrainBatches) },
    numTrainingIterations: 4,

    /* testing */
    testingBatchSize: 10000,
    randomTestingBatch () { 
        /* max: 10, batchsize: x */
        /* x=1000, 0<=start<=(max-1000) */
        const max = this.maxTestingSetSize - this.testingBatchSize
        const start = Math.floor(Math.random() * max)
        return [start, this.testingBatchSize]
    },

    /* model saving */

    snapshotDir: `file://${__dirname}/pretrained/`
}