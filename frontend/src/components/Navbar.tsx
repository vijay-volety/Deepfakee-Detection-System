import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const Navbar: React.FC = () => {
  const location = useLocation()

  const isActive = (path: string) => location.pathname === path

  return (
    <motion.nav 
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="bg-black bg-opacity-50 backdrop-blur-md p-4 sticky top-0 z-50"
    >
      <div className="container mx-auto flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold text-gradient">
          DeepFake Detection
        </Link>
        
        <div className="flex space-x-6">
          <Link 
            to="/" 
            className={`hover:text-blue-400 transition-colors ${
              isActive('/') ? 'text-blue-400' : 'text-white'
            }`}
          >
            Home
          </Link>
          <Link 
            to="/upload" 
            className={`hover:text-blue-400 transition-colors ${
              isActive('/upload') ? 'text-blue-400' : 'text-white'
            }`}
          >
            Upload
          </Link>
          <Link 
            to="/webcam" 
            className={`hover:text-blue-400 transition-colors ${
              isActive('/webcam') ? 'text-blue-400' : 'text-white'
            }`}
          >
            Webcam
          </Link>
          <Link 
            to="/admin" 
            className={`hover:text-blue-400 transition-colors ${
              isActive('/admin') ? 'text-blue-400' : 'text-white'
            }`}
          >
            Admin
          </Link>
        </div>
      </div>
    </motion.nav>
  )
}

export default Navbar