import React, { useState, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ArrowUpTrayIcon, 
  PhotoIcon, 
  VideoCameraIcon,
  InformationCircleIcon,
  DocumentArrowDownIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'

const UploadPage: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [consentGiven, setConsentGiven] = useState(false)
  const [videoDragActive, setVideoDragActive] = useState(false)
  const [imageDragActive, setImageDragActive] = useState(false)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      validateAndSetVideoFile(file)
    }
  }

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      validateAndSetImageFile(file)
    }
  }

  const validateAndSetVideoFile = (file: File) => {
    // Validate file type
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo']
    if (!validTypes.includes(file.type)) {
      setError('Please select a valid video file (MP4, MOV, AVI)')
      return
    }
    
    // Validate file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
      setError('File size exceeds 100MB limit')
      return
    }
    
    setVideoFile(file)
    setImageFile(null) // Clear image file if video is selected
    setError(null)
  }

  const validateAndSetImageFile = (file: File) => {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png']
    if (!validTypes.includes(file.type)) {
      setError('Please select a valid image file (JPG, PNG)')
      return
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size exceeds 10MB limit')
      return
    }
    
    setImageFile(file)
    setVideoFile(null) // Clear video file if image is selected
    setError(null)
  }

  const triggerVideoInput = () => {
    videoInputRef.current?.click()
  }

  const triggerImageInput = () => {
    imageInputRef.current?.click()
  }

  // Drag and drop handlers for video
  const handleVideoDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setVideoDragActive(true)
    } else if (e.type === "dragleave") {
      setVideoDragActive(false)
    }
  }

  const handleVideoDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setVideoDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      validateAndSetVideoFile(file)
    }
  }

  // Drag and drop handlers for image
  const handleImageDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setImageDragActive(true)
    } else if (e.type === "dragleave") {
      setImageDragActive(false)
    }
  }

  const handleImageDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setImageDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      validateAndSetImageFile(file)
    }
  }

  const uploadFile = async (file: File) => {
    setIsUploading(true)
    setUploadProgress(0)
    setError(null)
    
    try {
      // Create FormData
      const formData = new FormData()
      formData.append('file', file)
      formData.append('user_consent', consentGiven ? 'true' : 'false')
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)
      
      // Send request to backend
      const response = await fetch('/api/v1/detect', {
        method: 'POST',
        body: formData
      })
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }
      
      const data = await response.json()
      return data.job_id
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      throw err
    } finally {
      setIsUploading(false)
    }
  }

  const checkAnalysisStatus = async (jobId: string) => {
    setIsAnalyzing(true)
    
    // Poll for results every 2 seconds
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/result/${jobId}`)
        
        if (!response.ok) {
          throw new Error('Failed to fetch analysis results')
        }
        
        const data = await response.json()
        
        if (data.status === 'completed') {
          clearInterval(pollInterval)
          setAnalysisResult(data)
          setIsAnalyzing(false)
        } else if (data.status === 'failed') {
          clearInterval(pollInterval)
          setError(data.error_message || 'Analysis failed')
          setIsAnalyzing(false)
        }
      } catch (err) {
        clearInterval(pollInterval)
        setError(err instanceof Error ? err.message : 'Failed to check analysis status')
        setIsAnalyzing(false)
      }
    }, 2000)
  }

  const handleAnalyze = async () => {
    const file = videoFile || imageFile
    
    if (!file) {
      setError('Please select a file to analyze')
      return
    }
    
    if (!consentGiven) {
      setError('Please provide consent to proceed with analysis')
      return
    }
    
    try {
      const jobId = await uploadFile(file)
      await checkAnalysisStatus(jobId)
    } catch (err) {
      // Error already handled in uploadFile
    }
  }

  const downloadReport = async () => {
    if (!analysisResult) return
    
    try {
      const response = await fetch(`/api/v1/download/${analysisResult.job_id}/report`)
      
      if (!response.ok) {
        throw new Error('Failed to download report')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `deepfake_report_${analysisResult.job_id}.txt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download report')
    }
  }

  const resetAnalysis = () => {
    setVideoFile(null)
    setImageFile(null)
    setAnalysisResult(null)
    setError(null)
    setIsAnalyzing(false)
    setIsUploading(false)
    setConsentGiven(false)
    setVideoDragActive(false)
    setImageDragActive(false)
  }

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
              Upload Media for Analysis
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300 mb-8"
            >
              Upload videos or images to detect deepfake content using our AI system
            </motion.p>
          </div>

          {analysisResult ? (
            // Results View
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="card mb-8"
            >
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-white mb-2">Analysis Complete</h2>
                <p className="text-gray-400">Your file has been analyzed for deepfake content</p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-8 mb-8">
                <div className="text-center">
                  <div className="text-5xl font-bold text-green-400 mb-2">
                    {analysisResult.result?.authentic_percentage?.toFixed(1)}%
                  </div>
                  <div className="text-gray-400">Authentic Content</div>
                </div>
                
                <div className="text-center">
                  <div className="text-5xl font-bold text-red-400 mb-2">
                    {analysisResult.result?.deepfake_percentage?.toFixed(1)}%
                  </div>
                  <div className="text-gray-400">Deepfake Content</div>
                </div>
              </div>
              
              <div className="mb-8">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-300">Overall Confidence</span>
                    <span className="text-white font-medium">
                      {(analysisResult.result?.overall_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${analysisResult.result?.overall_confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-6">
                <button 
                  onClick={() => downloadReport()}
                  className="btn-accent px-6 py-3 inline-flex items-center gap-2"
                >
                  <DocumentArrowDownIcon className="w-5 h-5" />
                  Download Text Report
                </button>
              </div>
              
              <div className="text-center">
                <button 
                  onClick={resetAnalysis}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Analyze Another File
                </button>
              </div>
            </motion.div>
          ) : (
            // Upload View
            <>
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg mb-6 flex items-start gap-3"
                >
                  <XCircleIcon className="w-6 h-6 flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </motion.div>
              )}
              
              {(isUploading || isAnalyzing) && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="card mb-8"
                >
                  <div className="text-center">
                    <div className="flex justify-center mb-4">
                      <ArrowPathIcon className="w-12 h-12 text-blue-400 animate-spin" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      {isUploading ? 'Uploading File...' : 'Analyzing Content...'}
                    </h3>
                    <p className="text-gray-400 mb-4">
                      {isUploading 
                        ? 'Please wait while your file is being uploaded' 
                        : 'Our AI is analyzing your content for deepfake indicators'}
                    </p>
                    <div className="w-full bg-gray-700 rounded-full h-2.5 mb-2">
                      <div 
                        className="bg-blue-600 h-2.5 rounded-full" 
                        style={{ width: `${uploadProgress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-500">{uploadProgress}% complete</p>
                  </div>
                </motion.div>
              )}
              
              {/* Upload Options */}
              <motion.div 
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="grid md:grid-cols-2 gap-8 mb-8"
              >
                {/* Video Upload */}
                <div className="card">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                      <VideoCameraIcon className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">Video Analysis</h2>
                  </div>
                  
                  <p className="text-gray-400 mb-6">
                    Upload MP4 or MOV files for comprehensive deepfake detection. Our system will analyze 
                    each frame and provide detailed results.
                  </p>
                  
                  <div 
                    className={`border-2 border-dashed rounded-lg p-8 text-center mb-6 cursor-pointer transition-colors ${
                      videoDragActive 
                        ? 'border-blue-500 bg-blue-500/10' 
                        : 'border-gray-700 hover:border-blue-500'
                    }`}
                    onDragEnter={handleVideoDrag}
                    onDragOver={handleVideoDrag}
                    onDragLeave={handleVideoDrag}
                    onDrop={handleVideoDrop}
                    onClick={triggerVideoInput}
                  >
                    <ArrowUpTrayIcon className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400 mb-4">
                      {videoFile ? videoFile.name : 'Drag & drop video files here'}
                    </p>
                    <button 
                      type="button"
                      className="btn-primary px-6 py-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        triggerVideoInput()
                      }}
                    >
                      Select Video Files
                    </button>
                    <input
                      ref={videoInputRef}
                      type="file"
                      accept="video/mp4,video/quicktime,video/x-msvideo"
                      className="hidden"
                      onChange={handleVideoSelect}
                    />
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    <p>Supported formats: MP4, MOV, AVI</p>
                    <p>Maximum file size: 100MB</p>
                  </div>
                </div>

                {/* Image Upload */}
                <div className="card">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center">
                      <PhotoIcon className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">Image Analysis</h2>
                  </div>
                  
                  <p className="text-gray-400 mb-6">
                    Upload JPG or PNG images for deepfake detection. Our system will analyze facial 
                    features and provide authenticity scores.
                  </p>
                  
                  <div 
                    className={`border-2 border-dashed rounded-lg p-8 text-center mb-6 cursor-pointer transition-colors ${
                      imageDragActive 
                        ? 'border-green-500 bg-green-500/10' 
                        : 'border-gray-700 hover:border-green-500'
                    }`}
                    onDragEnter={handleImageDrag}
                    onDragOver={handleImageDrag}
                    onDragLeave={handleImageDrag}
                    onDrop={handleImageDrop}
                    onClick={triggerImageInput}
                  >
                    <ArrowUpTrayIcon className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400 mb-4">
                      {imageFile ? imageFile.name : 'Drag & drop image files here'}
                    </p>
                    <button 
                      type="button"
                      className="btn-primary px-6 py-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        triggerImageInput()
                      }}
                    >
                      Select Image Files
                    </button>
                    <input
                      ref={imageInputRef}
                      type="file"
                      accept="image/jpeg,image/png"
                      className="hidden"
                      onChange={handleImageSelect}
                    />
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    <p>Supported formats: JPG, PNG</p>
                    <p>Maximum file size: 10MB</p>
                  </div>
                </div>
              </motion.div>
              
              {/* Consent Checkbox */}
              {(videoFile || imageFile) && !isUploading && !isAnalyzing && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="card mb-8"
                >
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="consent-checkbox"
                      checked={consentGiven}
                      onChange={(e) => setConsentGiven(e.target.checked)}
                      className="mt-1 h-5 w-5 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="consent-checkbox" className="text-gray-300">
                      I consent to the analysis of my media file for deepfake detection. 
                      I understand that this file will be processed by our AI system and 
                      will be automatically deleted after analysis.
                    </label>
                  </div>
                </motion.div>
              )}
              
              {/* Action Button */}
              {(videoFile || imageFile) && !isUploading && !isAnalyzing && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-center mb-8"
                >
                  <button
                    onClick={handleAnalyze}
                    disabled={!consentGiven}
                    className={`px-8 py-4 text-lg inline-flex items-center gap-3 rounded-lg font-medium transition-colors ${
                      consentGiven 
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white' 
                        : 'bg-gray-700 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    <CheckCircleIcon className="w-6 h-6" />
                    Analyze for Deepfakes
                  </button>
                  {!consentGiven && (
                    <p className="text-red-400 text-sm mt-2">
                      Please provide consent to proceed with analysis
                    </p>
                  )}
                </motion.div>
              )}
            </>
          )}

          {/* Information Section */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-lg p-6 mb-8"
          >
            <div className="flex items-start gap-4">
              <InformationCircleIcon className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">How It Works</h3>
                <p className="text-gray-400">
                  Our AI system uses ResNet for feature extraction and LSTM for temporal analysis. 
                  After uploading your media, the system will process it and generate a detailed 
                  report with authenticity scores and visual explanations.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Navigation */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link to="/" className="btn-secondary px-6 py-3 text-center">
              Back to Home
            </Link>
            <Link to="/webcam" className="btn-primary px-6 py-3 text-center">
              Try Webcam Detection
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default UploadPage