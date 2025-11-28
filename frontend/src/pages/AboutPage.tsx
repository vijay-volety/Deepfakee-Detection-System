import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  InformationCircleIcon, 
  LightBulbIcon,
  ShieldCheckIcon,
  AcademicCapIcon
} from '@heroicons/react/24/outline'

const AboutPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="container mx-auto px-4 py-8"
      >
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-4xl font-bold text-white mb-4"
            >
              About DeepFake Detection
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300"
            >
              Understanding the technology behind our AI-powered deepfake detection system
            </motion.p>
          </div>

          {/* Introduction */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                <InformationCircleIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">What are Deepfakes?</h2>
            </div>
            
            <p className="text-gray-400 mb-4">
              Deepfakes are synthetic media in which a person in an existing image or video is replaced 
              with someone else's likeness using artificial neural networks. They can be used to create 
              convincing but fake videos of real people saying or doing things they never actually did.
            </p>
            
            <p className="text-gray-400 mb-4">
              Our system uses advanced machine learning techniques to detect these manipulations and 
              help verify the authenticity of digital media.
            </p>
          </motion.div>

          {/* Technology */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center">
                <LightBulbIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Our Technology</h2>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">ResNet Architecture</h3>
                <p className="text-gray-400">
                  We use a deep residual network to extract detailed facial features and identify 
                  inconsistencies that indicate manipulation.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">LSTM Analysis</h3>
                <p className="text-gray-400">
                  Long Short-Term Memory networks analyze temporal coherence across video frames 
                  to detect anomalies in facial movements.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Attention Mechanisms</h3>
                <p className="text-gray-400">
                  Self-attention models focus on critical facial regions to identify subtle signs 
                  of digital manipulation.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Ensemble Methods</h3>
                <p className="text-gray-400">
                  Multiple models work together to provide robust detection with high confidence 
                  scores and detailed explanations.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Privacy & Ethics */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
                <ShieldCheckIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Privacy & Ethics</h2>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Privacy First</h3>
                <p className="text-gray-400">
                  We do not store uploaded media by default. All processing happens in real-time 
                  and files are automatically deleted after analysis.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Ethical AI</h3>
                <p className="text-gray-400">
                  Our system is designed to help verify authenticity, not to create or distribute 
                  synthetic media. We're committed to responsible AI development.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Research */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-yellow-600 rounded-full flex items-center justify-center">
                <AcademicCapIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Research & Development</h2>
            </div>
            
            <p className="text-gray-400 mb-4">
              Our system is trained on multiple standard datasets including DFDC, FaceForensics++, 
              and Celeb-DF. We continuously update our models with the latest research findings 
              to maintain high detection accuracy.
            </p>
            
            <p className="text-gray-400">
              For more information about our research and technical details, please refer to our 
              documentation and academic publications.
            </p>
          </motion.div>

          {/* Navigation */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link to="/" className="btn-secondary px-6 py-3 text-center">
              Back to Home
            </Link>
            <Link to="/privacy" className="btn-primary px-6 py-3 text-center">
              Privacy Policy
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default AboutPage