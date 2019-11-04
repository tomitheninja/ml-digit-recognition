const debug = require('debug')('loader')

const app = require('./app')

const IS_NODEJS = typeof(window) === 'undefined' && typeof(global) === 'object'

if (IS_NODEJS) {
    debug('Running in NodeJS')

    if (!process.env.TF_DISABLE_NODE) {
        
        
        if (process.env.TF_ENABLE_GPU) {
            debug('Enabled GPU acceleration')
            require('@tensorflow/tfjs-node-gpu')
        } else {
            debug('Enabled CPU acceleration')
            require('@tensorflow/tfjs-node')
        }
    }
} else {
    debug('Running in browser??')
}

const tf = require('@tensorflow/tfjs')
debug('TensorFlow loaded - backend: ' + tf.getBackend())

app()
