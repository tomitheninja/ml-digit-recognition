const mnist = require('mnist-data')
const _ = require('lodash')
const debug = require('debug')('MNIST-database')

const config = require('../config')

module.exports = function (type = 'training', start, num) {
    const end = start + num

    let db
    switch (type) {
        case 'training':
            db = mnist.training
            if (end > config.maxTrainingSetSize) {
                throw new Error('Out of MNIST data range; start=' + start +  ', end=' + end)
            }
            debug(`Downloaded mnist.training of range(${start}, ${end})`)
            break
        case 'testing':
            db = mnist.testing
            if (end > config.maxTestingSetSize) {
                throw new Error('Out of MNIST data range; start=' + start +  ', end=' + end)
            }
            debug(`Downloaded mnist.testing set of range(${start}, ${end})`)
            break
        default:
            throw new Error('Unknown MNIST db type ' + type);
    }
    const { images, labels } = db(start, end)
    debug('Starting to parse data')
    const result = [_.flattenDeep(images.values), labels.values]
    debug('done')
    return result
}