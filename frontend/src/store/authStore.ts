import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  isAuthenticated: boolean
  user: {
    id: string
    username: string
    email: string
    isAdmin: boolean
  } | null
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
  checkAuthStatus: () => Promise<boolean>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      user: null,
      
      login: async (username: string, password: string) => {
        // In a real application, this would be an API call
        // For demo purposes, we'll simulate a successful login
        if (username && password) {
          set({
            isAuthenticated: true,
            user: {
              id: '1',
              username,
              email: `${username}@deepfakedetection.com`,
              isAdmin: username === 'admin'
            }
          })
          return true
        }
        return false
      },
      
      logout: () => {
        set({
          isAuthenticated: false,
          user: null
        })
      },
      
      checkAuthStatus: async () => {
        // In a real application, this would check with the backend
        // For now, we'll just return the current state
        return get().isAuthenticated
      }
    }),
    {
      name: 'auth-storage', // name of the item in the storage (must be unique)
      partialize: (state) => ({ 
        isAuthenticated: state.isAuthenticated,
        user: state.user
      }),
    }
  )
)