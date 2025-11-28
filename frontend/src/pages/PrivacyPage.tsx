import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ShieldCheckIcon, 
  DocumentTextIcon,
  ClockIcon,
  TrashIcon
} from '@heroicons/react/24/outline'

const PrivacyPage: React.FC = () => {
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
              Privacy Policy
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300"
            >
              How we collect, use, and protect your information
            </motion.p>
          </div>

          {/* Last Updated */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                <ClockIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Last Updated</h2>
                <p className="text-gray-400">October 27, 2025</p>
              </div>
            </div>
          </motion.div>

          {/* Information Collection */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center">
                <DocumentTextIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Information We Collect</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Media Files</h3>
                <p className="text-gray-400">
                  When you upload videos or images for analysis, we temporarily process these files 
                  to perform deepfake detection. By default, we do not store these files permanently.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Usage Data</h3>
                <p className="text-gray-400">
                  We may collect information about how you interact with our service, including 
                  features used, time spent, and error logs to improve our system.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Device Information</h3>
                <p className="text-gray-400">
                  We may collect information about the device you use to access our service, 
                  including browser type, operating system, and IP address for security purposes.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Data Usage */}
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
              <h2 className="text-2xl font-bold text-white">How We Use Your Information</h2>
            </div>
            
            <ul className="space-y-3">
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-purple-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                </div>
                <span className="text-gray-300">To provide and improve our deepfake detection services</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-purple-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                </div>
                <span className="text-gray-300">To monitor and analyze usage patterns for system optimization</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-purple-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                </div>
                <span className="text-gray-300">To detect, prevent, and address technical issues</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-purple-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                </div>
                <span className="text-gray-300">To communicate with you about updates and important notices</span>
              </li>
            </ul>
          </motion.div>

          {/* Data Protection */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-red-600 rounded-full flex items-center justify-center">
                <TrashIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Data Retention & Deletion</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Automatic Deletion</h3>
                <p className="text-gray-400">
                  Uploaded media files are automatically deleted within 24 hours of upload. 
                  Analysis results are retained for 30 days to allow access to reports.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Manual Deletion</h3>
                <p className="text-gray-400">
                  You can request deletion of your data at any time by contacting our support team. 
                  We will process such requests within 30 days.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Data Security</h3>
                <p className="text-gray-400">
                  We implement appropriate security measures to protect your information, 
                  including encryption in transit and at rest where applicable.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Your Rights */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="card mb-8"
          >
            <h2 className="text-2xl font-bold text-white mb-6">Your Privacy Rights</h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Access & Correction</h3>
                <p className="text-gray-400">
                  You have the right to access and correct your personal information held by us.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Data Portability</h3>
                <p className="text-gray-400">
                  You can request a copy of your data in a structured, machine-readable format.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Objection</h3>
                <p className="text-gray-400">
                  You may object to the processing of your personal data in certain circumstances.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Complaints</h3>
                <p className="text-gray-400">
                  You have the right to lodge a complaint with a supervisory authority if you 
                  believe we have violated your privacy rights.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Contact */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="card mb-8"
          >
            <h2 className="text-2xl font-bold text-white mb-6">Contact Us</h2>
            
            <p className="text-gray-400 mb-4">
              If you have any questions about this Privacy Policy or our privacy practices, 
              please contact us at:
            </p>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-white">privacy@deepfakedetection.com</p>
            </div>
          </motion.div>

          {/* Navigation */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link to="/" className="btn-secondary px-6 py-3 text-center">
              Back to Home
            </Link>
            <Link to="/about" className="btn-primary px-6 py-3 text-center">
              About DeepFakes
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default PrivacyPage