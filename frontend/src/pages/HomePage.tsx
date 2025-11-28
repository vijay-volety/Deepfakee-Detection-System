import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  PlayIcon, 
  CameraIcon, 
  ShieldCheckIcon, 
  EyeIcon,
  DocumentChartBarIcon,
  CogIcon
} from '@heroicons/react/24/outline'

const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="container mx-auto px-4 py-16"
      >
        <div className="text-center mb-16">
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="text-6xl font-bold text-white mb-6"
          >
            <span className="text-gradient">DeepFake</span> Detection System
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            AI-powered detection system that analyzes videos and images to identify deepfake content 
            with state-of-the-art ResNet + LSTM architecture
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link 
              to="/upload" 
              className="btn-primary px-8 py-4 text-lg inline-flex items-center gap-3"
            >
              <PlayIcon className="w-6 h-6" />
              Upload Video/Image
            </Link>
            
            <Link 
              to="/webcam" 
              className="btn-secondary px-8 py-4 text-lg inline-flex items-center gap-3"
            >
              <CameraIcon className="w-6 h-6" />
              Live Webcam Detection
            </Link>
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.8 }}
          className="grid md:grid-cols-3 gap-8 mb-16"
        >
          <div className="card text-center">
            <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <ShieldCheckIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Accurate Detection</h3>
            <p className="text-gray-400">
              Advanced ResNet + LSTM model achieves 85%+ accuracy on standard deepfake datasets
            </p>
          </div>

          <div className="card text-center">
            <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <EyeIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Explainable AI</h3>
            <p className="text-gray-400">
              Visual saliency maps show exactly which regions triggered deepfake detection
            </p>
          </div>

          <div className="card text-center">
            <div className="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <DocumentChartBarIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Detailed Reports</h3>
            <p className="text-gray-400">
              Comprehensive analysis with per-frame scores, confidence intervals, and downloadable reports
            </p>
          </div>
        </motion.div>

        {/* Stats Section */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.8 }}
          className="glass rounded-lg p-8 mb-16"
        >
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl font-bold text-primary-400 mb-2">85%+</div>
              <div className="text-gray-300">Detection Accuracy</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-400 mb-2">&lt;30s</div>
              <div className="text-gray-300">Processing Time</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-400 mb-2">3</div>
              <div className="text-gray-300">Training Datasets</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-400 mb-2">100MB</div>
              <div className="text-gray-300">Max File Size</div>
            </div>
          </div>
        </motion.div>

        {/* How It Works */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.8 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-8">How It Works</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-lg mb-4">
                1
              </div>
              <h4 className="font-semibold text-white mb-2">Upload Media</h4>
              <p className="text-gray-400 text-sm">Upload video or image files, or capture from webcam</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center text-white font-bold text-lg mb-4">
                2
              </div>
              <h4 className="font-semibold text-white mb-2">AI Analysis</h4>
              <p className="text-gray-400 text-sm">ResNet extracts features, LSTM analyzes temporal patterns</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold text-lg mb-4">
                3
              </div>
              <h4 className="font-semibold text-white mb-2">Detection</h4>
              <p className="text-gray-400 text-sm">AI model predicts authentic vs deepfake probability</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-12 h-12 bg-red-600 rounded-full flex items-center justify-center text-white font-bold text-lg mb-4">
                4
              </div>
              <h4 className="font-semibold text-white mb-2">Results</h4>
              <p className="text-gray-400 text-sm">Get detailed report with explanations and confidence scores</p>
            </div>
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.8 }}
          className="text-center mt-16"
        >
          <h2 className="text-2xl font-bold text-white mb-4">Ready to Detect Deepfakes?</h2>
          <p className="text-gray-300 mb-8">Start analyzing your media files today</p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/upload" className="btn-primary px-6 py-3">
              Get Started
            </Link>
            <Link to="/about" className="btn-secondary px-6 py-3">
              Learn More
            </Link>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default HomePage