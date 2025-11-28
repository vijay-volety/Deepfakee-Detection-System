import React from 'react'
import { Navigate } from 'react-router-dom'

interface ProtectedRouteProps {
  children: React.ReactNode
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  // In a real application, you would check if the user is authenticated
  // For now, we'll simulate this with a simple check
  const isAuthenticated = false // Replace with actual authentication check

  if (!isAuthenticated) {
    // Redirect to login page if not authenticated
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

export default ProtectedRoute