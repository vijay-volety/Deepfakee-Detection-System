import React from 'react'
import { Link, Outlet } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  CogIcon, 
  ChartBarIcon,
  DocumentTextIcon,
  ArrowPathIcon,
  UserIcon,
  ServerIcon
} from '@heroicons/react/24/outline'

const AdminPage: React.FC = () => {
  const navItems = [
    { name: 'Dashboard', path: '', icon: ChartBarIcon },
    { name: 'Models', path: 'models', icon: ServerIcon },
    { name: 'Logs', path: 'logs', icon: DocumentTextIcon },
    { name: 'Retraining', path: 'retrain', icon: ArrowPathIcon },
    { name: 'Users', path: 'users', icon: UserIcon },
  ]

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
              Admin Dashboard
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-300"
            >
              Manage the DeepFake Detection System
            </motion.p>
          </div>

          <div className="flex flex-col lg:flex-row gap-8">
            {/* Sidebar Navigation */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="lg:w-64 flex-shrink-0"
            >
              <div className="card">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                    <CogIcon className="w-6 h-6 text-white" />
                  </div>
                  <h2 className="text-xl font-bold text-white">Admin Panel</h2>
                </div>
                
                <nav className="space-y-2">
                  {navItems.map((item, index) => {
                    const Icon = item.icon
                    return (
                      <Link
                        key={index}
                        to={item.path}
                        className="flex items-center gap-3 w-full p-3 rounded-lg text-gray-300 hover:bg-gray-800 hover:text-white transition-colors"
                      >
                        <Icon className="w-5 h-5" />
                        <span>{item.name}</span>
                      </Link>
                    )
                  })}
                </nav>
              </div>
            </motion.div>

            {/* Main Content Area */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="flex-1"
            >
              <div className="card">
                <Outlet />
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default AdminPage