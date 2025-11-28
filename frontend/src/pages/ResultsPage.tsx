import React from 'react'
import { Link, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ChartBarIcon, 
  PhotoIcon,
  DocumentArrowDownIcon,
  InformationCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'

const ResultsPage: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>()
  
  // Mock data for demonstration
  const resultData = {
    jobId: jobId || 'abc123',
    fileName: 'sample-video.mp4',
    fileType: 'video',
    status: 'completed',
    authenticityScore: 87.5,
    isDeepfake: false,
    processingTime: '24.3s',
    framesAnalyzed: 142,
    createdAt: new Date().toLocaleString(),
    confidence: 'High',
    recommendations: [
      'No deepfake indicators detected',
      'Facial features appear natural',
      'Lighting and shadows are consistent'
    ]
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="container mx-auto px-4 py-8"
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-4xl font-bold text-white mb-4"
            >
              Analysis Results
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300"
            >
              Detailed report for <span className="text-primary-400">{resultData.fileName}</span>
            </motion.p>
          </div>

          {/* Summary Card */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card mb-8"
          >
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-5xl font-bold text-green-400 mb-2">
                  {resultData.authenticityScore}%
                </div>
                <div className="text-gray-400">Authenticity Score</div>
                <div className="mt-2">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    resultData.isDeepfake 
                      ? 'bg-red-900 text-red-300' 
                      : 'bg-green-900 text-green-300'
                  }`}>
                    {resultData.isDeepfake ? 'DEEPFAKE' : 'AUTHENTIC'}
                  </span>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-5xl font-bold text-blue-400 mb-2">
                  {resultData.framesAnalyzed}
                </div>
                <div className="text-gray-400">Frames Analyzed</div>
                <div className="mt-2">
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-blue-900 text-blue-300">
                    {resultData.confidence} Confidence
                  </span>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-5xl font-bold text-purple-400 mb-2">
                  {resultData.processingTime}
                </div>
                <div className="text-gray-400">Processing Time</div>
                <div className="mt-2">
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-purple-900 text-purple-300">
                    Completed
                  </span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Detailed Results */}
          <div className="grid lg:grid-cols-3 gap-8 mb-8">
            {/* Visual Preview */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="lg:col-span-2 card"
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                  <PhotoIcon className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Visual Analysis</h2>
              </div>
              
              <div className="bg-gray-900 rounded-lg overflow-hidden mb-6">
                <div className="aspect-video bg-gray-800 flex items-center justify-center">
                  <div className="text-center">
                    <PhotoIcon className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-500">Visual analysis heatmap would appear here</p>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-4">
                <button className="btn-primary px-6 py-2 inline-flex items-center gap-2">
                  <ArrowPathIcon className="w-5 h-5" />
                  Re-analyze
                </button>
              </div>
            </motion.div>

            {/* Recommendations */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="card"
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center">
                  <InformationCircleIcon className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Recommendations</h2>
              </div>
              
              <ul className="space-y-4 mb-6">
                {resultData.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-green-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-green-400"></div>
                    </div>
                    <span className="text-gray-300">{rec}</span>
                  </li>
                ))}
              </ul>
              
              <div className="pt-4 border-t border-gray-800">
                <h3 className="font-semibold text-white mb-3">Download Report</h3>
                <div className="flex flex-wrap gap-3">
                  <button className="btn-secondary px-4 py-2 inline-flex items-center gap-2">
                    <DocumentArrowDownIcon className="w-5 h-5" />
                    PDF Report
                  </button>
                  <button className="btn-secondary px-4 py-2 inline-flex items-center gap-2">
                    <DocumentArrowDownIcon className="w-5 h-5" />
                    JSON Data
                  </button>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Metadata */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card mb-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
                <ChartBarIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Analysis Metadata</h2>
            </div>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Job ID</div>
                <div className="text-white font-mono">{resultData.jobId}</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">File Name</div>
                <div className="text-white">{resultData.fileName}</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">File Type</div>
                <div className="text-white capitalize">{resultData.fileType}</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Created At</div>
                <div className="text-white">{resultData.createdAt}</div>
              </div>
            </div>
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
            <Link to="/upload" className="btn-primary px-6 py-3 text-center">
              Analyze Another File
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default ResultsPage