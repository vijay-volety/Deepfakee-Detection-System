import React, { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import Webcam from 'react-webcam'
import { 
  CameraIcon, 
  VideoCameraIcon,
  PlayIcon,
  StopIcon,
  InformationCircleIcon,
  ArrowPathIcon,
  DocumentArrowDownIcon
} from '@heroicons/react/24/outline'

const WebcamPage: React.FC = () => {
  const webcamRef = useRef<Webcam>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([])
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [consentGiven, setConsentGiven] = useState(false)
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null)

  // Check camera permission on component mount
  useEffect(() => {
    checkCameraPermission()
  }, [])

  const checkCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setHasCameraPermission(true)
      stream.getTracks().forEach(track => track.stop())
    } catch (err) {
      setHasCameraPermission(false)
      setError('Camera permission denied. Please allow camera access to use this feature.')
    }
  }

  const startRecording = () => {
    if (!consentGiven) {
      setError('Please provide consent to proceed with recording')
      return
    }
    
    setError(null)
    setRecordedChunks([])
    
    if (webcamRef.current && webcamRef.current.stream) {
      const stream = webcamRef.current.stream
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' })
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data])
        }
      }
      
      mediaRecorderRef.current.start()
      setIsRecording(true)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const analyzeFrames = async () => {
    if (recordedChunks.length === 0) {
      setError('No recorded video to analyze. Please record some footage first.')
      return
    }
    
    if (!consentGiven) {
      setError('Please provide consent to proceed with analysis')
      return
    }
    
    setIsAnalyzing(true)
    setError(null)
    
    try {
      // Create video blob from recorded chunks
      const blob = new Blob(recordedChunks, { type: 'video/webm' })
      const file = new File([blob], 'webcam_recording.webm', { type: 'video/webm' })
      
      // Upload for analysis
      const jobId = await uploadFile(file)
      await checkAnalysisStatus(jobId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setIsAnalyzing(false)
    }
  }

  const uploadFile = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('user_consent', consentGiven ? 'true' : 'false')
    
    const response = await fetch('/api/v1/detect', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`)
    }
    
    const data = await response.json()
    return data.job_id
  }

  const checkAnalysisStatus = async (jobId: string) => {
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
    setAnalysisResult(null)
    setRecordedChunks([])
    setError(null)
    setIsAnalyzing(false)
    setIsRecording(false)
  }

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
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
              Live Webcam Detection
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300 mb-8"
            >
              Capture live video from your webcam for real-time deepfake detection
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
                <p className="text-gray-400">Your webcam recording has been analyzed for deepfake content</p>
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
                  Analyze Another Recording
                </button>
              </div>
            </motion.div>
          ) : (
            <>
              {/* Consent Checkbox */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="card mb-6"
              >
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="consent"
                    checked={consentGiven}
                    onChange={(e) => setConsentGiven(e.target.checked)}
                    className="mt-1 h-5 w-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                  <label htmlFor="consent" className="text-gray-300">
                    I consent to the analysis of my webcam recording for deepfake detection purposes. 
                    I understand that no personal data will be stored permanently.
                  </label>
                </div>
              </motion.div>

              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg mb-6 flex items-start gap-3"
                >
                  <InformationCircleIcon className="w-6 h-6 flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </motion.div>
              )}

              {isAnalyzing && (
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
                      Analyzing Content...
                    </h3>
                    <p className="text-gray-400">
                      Our AI is analyzing your webcam recording for deepfake indicators
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Webcam Preview */}
              <motion.div 
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card mb-8"
              >
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
                    <VideoCameraIcon className="w-6 h-6 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold text-white">Webcam Feed</h2>
                </div>
                
                <div className="bg-gray-900 rounded-lg overflow-hidden mb-6">
                  {hasCameraPermission === false ? (
                    <div className="aspect-video bg-gray-800 flex flex-col items-center justify-center p-4 text-center">
                      <CameraIcon className="w-16 h-16 text-gray-600 mb-4" />
                      <p className="text-gray-400 mb-4">
                        Camera permission is required to use this feature. Please allow camera access in your browser settings.
                      </p>
                      <button 
                        onClick={checkCameraPermission}
                        className="btn-primary px-4 py-2"
                      >
                        Retry Camera Permission
                      </button>
                    </div>
                  ) : (
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      videoConstraints={videoConstraints}
                      className="w-full aspect-video object-cover"
                      screenshotFormat="image/jpeg"
                    />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-4 justify-center">
                  {!isRecording ? (
                    <button 
                      onClick={startRecording}
                      disabled={!consentGiven || isAnalyzing}
                      className={`px-6 py-3 inline-flex items-center gap-2 ${
                        !consentGiven || isAnalyzing 
                          ? 'bg-gray-600 cursor-not-allowed' 
                          : 'btn-primary hover:opacity-90'
                      }`}
                    >
                      <PlayIcon className="w-5 h-5" />
                      Start Recording
                    </button>
                  ) : (
                    <button 
                      onClick={stopRecording}
                      className="btn-secondary px-6 py-3 inline-flex items-center gap-2"
                    >
                      <StopIcon className="w-5 h-5" />
                      Stop Recording
                    </button>
                  )}
                  
                  <button 
                    onClick={analyzeFrames}
                    disabled={recordedChunks.length === 0 || isAnalyzing}
                    className={`px-6 py-3 ${
                      recordedChunks.length === 0 || isAnalyzing
                        ? 'bg-gray-600 cursor-not-allowed' 
                        : 'btn-accent hover:opacity-90'
                    }`}
                  >
                    Analyze Frames
                  </button>
                </div>
                
                {recordedChunks.length > 0 && !isAnalyzing && (
                  <div className="mt-4 text-center text-gray-400">
                    Recorded {recordedChunks.length} chunks. Ready for analysis.
                  </div>
                )}
              </motion.div>
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
              <InformationCircleIcon className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Real-time Detection</h3>
                <p className="text-gray-400">
                  Our system captures video frames from your webcam and analyzes them in real-time 
                  for deepfake detection. Press "Start Recording" to begin capturing, and 
                  "Analyze Frames" to process the captured frames with our AI model.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Features */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="grid md:grid-cols-3 gap-6 mb-8"
          >
            <div className="card text-center">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <VideoCameraIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Live Capture</h3>
              <p className="text-gray-400 text-sm">
                Capture video directly from your webcam for immediate analysis
              </p>
            </div>
            
            <div className="card text-center">
              <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <PlayIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Real-time Processing</h3>
              <p className="text-gray-400 text-sm">
                Analyze frames as they are captured for instant feedback
              </p>
            </div>
            
            <div className="card text-center">
              <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <InformationCircleIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Detailed Reports</h3>
              <p className="text-gray-400 text-sm">
                Generate comprehensive reports with authenticity scores
              </p>
            </div>
          </motion.div>

          {/* Navigation */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link to="/" className="btn-secondary px-6 py-3 text-center">
              Back to Home
            </Link>
            <Link to="/upload" className="btn-primary px-6 py-3 text-center">
              Upload Files Instead
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default WebcamPage