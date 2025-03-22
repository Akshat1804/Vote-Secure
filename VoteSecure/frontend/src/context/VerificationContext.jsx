"use client"

import { createContext, useContext, useState, useEffect, useRef } from "react"

// Create the context
const VerificationContext = createContext()

// Custom hook to use the verification context
export const useVerification = () => {
  const context = useContext(VerificationContext)
  if (!context) {
    throw new Error("useVerification must be used within a VerificationProvider")
  }
  return context
}

export const VerificationProvider = ({ children }) => {
  // User data state
  const [userData, setUserData] = useState(null)

  // Verification steps state
  const [step, setStep] = useState(1) // 1: Mobile verification, 2: Identity verification
  const [numberVerified, setNumberVerified] = useState(false)
  const [faceVerified, setFaceVerified] = useState(false)
  const [secondaryVerificationComplete, setSecondaryVerificationComplete] = useState(false)

  // Mobile verification state
  const [verificationNumbers, setVerificationNumbers] = useState([])
  const [correctNumber, setCorrectNumber] = useState(null)

  // Face verification state
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [faceDetectionQuality, setFaceDetectionQuality] = useState(0)
  const [livenessDetected, setLivenessDetected] = useState(false)
  const [previousLandmarks, setPreviousLandmarks] = useState([])
  const [livenessChecks, setLivenessChecks] = useState(0)
  const [faceDetectionMode, setFaceDetectionMode] = useState("enhanced")
  const [faceVerificationStage, setFaceVerificationStage] = useState(0)
  const [captureCount, setCaptureCount] = useState(0)
  const [verificationProgress, setVerificationProgress] = useState(0)

  // Camera and models state
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [cameraPermission, setCameraPermission] = useState(null)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [modelLoadingError, setModelLoadingError] = useState(null)
  const [webGLSupported, setWebGLSupported] = useState(true)

  // UI state
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [alertMessage, setAlertMessage] = useState(null)
  const [alertVariant, setAlertVariant] = useState("info")
  const [showSecurityInfo, setShowSecurityInfo] = useState(false)
  const [showTroubleshooting, setShowTroubleshooting] = useState(false)
  const [showAlternativeMethod, setShowAlternativeMethod] = useState(false)

  // Security state
  const [verificationMethod, setVerificationMethod] = useState("facial")
  const [verificationAttempts, setVerificationAttempts] = useState(0)
  const [isLockedOut, setIsLockedOut] = useState(false)
  const [lockoutTime, setLockoutTime] = useState(0)
  const [sessionTimer, setSessionTimer] = useState(600) // 10 minutes
  const [sessionTimerActive, setSessionTimerActive] = useState(false)
  const [deviceFingerprint, setDeviceFingerprint] = useState("")
  const [ipAddress, setIpAddress] = useState("")
  const [geoLocation, setGeoLocation] = useState(null)

  // Add these state variables to the existing state in the VerificationProvider component
  const [facialRecognitionFailed, setFacialRecognitionFailed] = useState(false)
  const [facialAttempts, setFacialAttempts] = useState(0)
  const [showAlternativeOption, setShowAlternativeOption] = useState(false)

  // Refs
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const animationRef = useRef(null)
  const livenessIntervalRef = useRef(null)
  const lockoutTimerRef = useRef(null)
  const detectionInterval = useRef(null)

  // Format timer
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  // Show alert message
  const showAlert = (message, variant = "info") => {
    setAlertMessage(message)
    setAlertVariant(variant)
    setError(message)

    // Auto-hide success alerts after 5 seconds
    if (variant === "success") {
      setTimeout(() => {
        setAlertMessage(null)
        setError(null)
      }, 5000)
    }
  }

  // Handle failed verification attempt
  const handleFailedAttempt = () => {
    const newAttempts = verificationAttempts + 1
    setVerificationAttempts(newAttempts)

    // Implement progressive lockout
    if (newAttempts >= 5) {
      const lockoutDuration = Math.min(Math.pow(2, newAttempts - 5) * 30, 1800) // Exponential backoff, max 30 minutes
      setLockoutTime(lockoutDuration)
      setIsLockedOut(true)
      showAlert(`Too many failed attempts. Account locked for ${formatTime(lockoutDuration)}.`, "error")

      // Log security event
      console.log(
        `Security alert: Account locked due to ${newAttempts} failed verification attempts. Device: ${deviceFingerprint}, IP: ${ipAddress}`,
      )
    }
  }

  // Generate 3 random 2-digit numbers for verification
  const generateVerificationNumbers = () => {
    const numbers = []
    for (let i = 0; i < 3; i++) {
      numbers.push(Math.floor(10 + Math.random() * 90)) // 2-digit numbers (10-99)
    }
    setVerificationNumbers(numbers)

    // Set one of them as the correct number
    const correctIndex = Math.floor(Math.random() * 3)
    setCorrectNumber(numbers[correctIndex])

    // In a real app, this would send an SMS to the user's phone
    console.log(`Correct verification number: ${numbers[correctIndex]}`)
  }

  // Handle number verification
  const verifyNumber = (selectedNumber) => {
    if (isLockedOut) {
      showAlert(`Account is locked. Please try again in ${formatTime(lockoutTime)}.`, "error")
      return
    }

    if (selectedNumber === correctNumber) {
      setNumberVerified(true)
      setStep(2)
      showAlert("Mobile verification successful!", "success")
    } else {
      handleFailedAttempt()
      showAlert("Incorrect number selected. Please try again.", "error")
      // Generate new numbers
      generateVerificationNumbers()
    }
  }

  // Stop camera
  const stopCamera = () => {
    console.log("Stopping camera")

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        console.log("Stopping track:", track)
        track.stop()
      })
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsCameraActive(false)

    if (detectionInterval.current) {
      clearInterval(detectionInterval.current)
      detectionInterval.current = null
    }

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    if (livenessIntervalRef.current) {
      clearInterval(livenessIntervalRef.current)
      livenessIntervalRef.current = null
    }
  }

  // Handle secondary verification success
  const handleSecondaryVerificationSuccess = () => {
    setSecondaryVerificationComplete(true)
    setFaceVerified(true)
    showAlert("Verification successful! You can now proceed to voting.", "success")
  }

  // Handle secondary verification error
  const handleSecondaryVerificationError = (errorMessage) => {
    showAlert(errorMessage, "error")
  }

  // Toggle security info
  const toggleSecurityInfo = () => {
    setShowSecurityInfo((prev) => !prev)
  }

  // Toggle troubleshooting section
  const toggleTroubleshooting = () => {
    setShowTroubleshooting((prev) => !prev)
  }

  // Switch to alternative verification method
  const switchToAlternativeMethod = () => {
    // This function is now disabled
    console.log("Alternative verification method is disabled")
    return
  }

  // Add this function to the VerificationProvider component
  const handleFacialRecognitionFailure = () => {
    const newAttempts = facialAttempts + 1
    setFacialAttempts(newAttempts)

    // After 2 failed attempts, suggest the alternative method
    if (newAttempts >= 2) {
      setFacialRecognitionFailed(true)
      setShowAlternativeOption(true)
      showAlert(
        "Facial recognition is having trouble. You can try again or use an alternative verification method.",
        "warning",
      )
    } else {
      showAlert("Facial recognition failed. Please ensure good lighting and position your face clearly.", "error")
    }
  }

  // Add this function to the VerificationProvider component
  const switchToAlternativeVerification = () => {
    stopCamera()
    setShowAlternativeMethod(true)
    setVerificationMethod("alternative")
    showAlert("Switched to alternative verification method.", "info")

    // Log this event for security purposes
    console.log(`User switched to alternative verification. Device: ${deviceFingerprint}, IP: ${ipAddress}`)
  }

  // Load user data from session storage or use mock data for testing
  useEffect(() => {
    try {
      const storedUserData = sessionStorage.getItem("userData")

      if (storedUserData) {
        const parsedUserData = JSON.parse(storedUserData)
        setUserData(parsedUserData)
      } else {
        // For testing: Create mock user data if none exists
        const mockUserData = {
          name: "Rahul Sharma",
          aadhaarNumber: "1234 5678 9012",
          phoneNumber: "9876543210", // Ensure this is always set
          voterID: "ABC1234567",
          constituency: "Central Delhi",
          address: "123 Main Street, New Delhi",
          dateOfBirth: "1985-05-15",
          email: "rahul.sharma@example.com",
        }
        sessionStorage.setItem("userData", JSON.stringify(mockUserData))
        setUserData(mockUserData)
        console.log("Created mock user data for testing")
      }

      // Generate verification numbers
      generateVerificationNumbers()

      // Start with enhanced face detection by default - more reliable
      setFaceDetectionMode("enhanced")
      setModelsLoaded(true)

      // Start session timer
      setSessionTimerActive(true)
    } catch (error) {
      console.error("Error handling user data:", error)
      showAlert("Error loading user data. Please return to dashboard and try again.", "error")
    }
  }, [])

  // Session timer effect
  useEffect(() => {
    if (sessionTimerActive && sessionTimer > 0) {
      timerRef.current = setTimeout(() => {
        setSessionTimer((prev) => prev - 1)
      }, 1000)
    } else if (sessionTimer === 0) {
      // Session expired
      showAlert("Your session has expired for security reasons. Please refresh the page and try again.", "error")
      stopCamera()
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [sessionTimerActive, sessionTimer])

  // Lockout timer effect
  useEffect(() => {
    if (isLockedOut && lockoutTime > 0) {
      lockoutTimerRef.current = setTimeout(() => {
        setLockoutTime((prev) => prev - 1)
      }, 1000)
    } else if (lockoutTime === 0 && isLockedOut) {
      setIsLockedOut(false)
      setVerificationAttempts(0)
      showAlert("Your account has been unlocked. You can try again now.", "info")
    }

    return () => {
      if (lockoutTimerRef.current) clearTimeout(lockoutTimerRef.current)
    }
  }, [isLockedOut, lockoutTime])

  // Generate a device fingerprint
  useEffect(() => {
    const generateFingerprint = () => {
      const userAgent = navigator.userAgent
      const screenWidth = window.screen.width
      const screenHeight = window.screen.height
      const colorDepth = window.screen.colorDepth
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
      const language = navigator.language

      // Create a simple hash of these values
      const fingerprintString = `${userAgent}|${screenWidth}x${screenHeight}|${colorDepth}|${timezone}|${language}`
      const hash = Array.from(fingerprintString)
        .reduce((hash, char) => (hash << 5) - hash + char.charCodeAt(0), 0)
        .toString(36)

      setDeviceFingerprint(hash)
    }

    // Get IP address (in a real app, this would be done server-side)
    const getIpAddress = async () => {
      try {
        const response = await fetch("https://api.ipify.org?format=json")
        const data = await response.json()
        setIpAddress(data.ip)
      } catch (error) {
        console.error("Error fetching IP:", error)
        setIpAddress("Unknown")
      }
    }

    // Get geolocation if available
    const getGeolocation = () => {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setGeoLocation({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
            })
          },
          (error) => {
            console.error("Geolocation error:", error)
          },
          { timeout: 10000 },
        )
      }
    }

    generateFingerprint()
    getIpAddress()
    getGeolocation()
  }, [])

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera()
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current)
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (livenessIntervalRef.current) {
        clearInterval(livenessIntervalRef.current)
      }
      if (lockoutTimerRef.current) {
        clearTimeout(lockoutTimerRef.current)
      }
    }
  }, [])

  // Context value
  const value = {
    // State
    userData,
    step,
    numberVerified,
    faceVerified,
    secondaryVerificationComplete,
    verificationNumbers,
    correctNumber,
    isCameraActive,
    faceDetected,
    faceDetectionQuality,
    livenessDetected,
    previousLandmarks,
    livenessChecks,
    faceDetectionMode,
    faceVerificationStage,
    captureCount,
    verificationProgress,
    modelsLoaded,
    cameraPermission,
    isModelLoading,
    modelLoadingError,
    webGLSupported,
    isLoading,
    error,
    alertMessage,
    alertVariant,
    showSecurityInfo,
    showTroubleshooting,
    showAlternativeMethod,
    verificationMethod,
    verificationAttempts,
    isLockedOut,
    lockoutTime,
    sessionTimer,
    sessionTimerActive,
    deviceFingerprint,
    ipAddress,
    geoLocation,
    facialRecognitionFailed,
    facialAttempts,
    showAlternativeOption,

    // Refs
    videoRef,
    canvasRef,
    streamRef,

    // Functions
    setStep,
    setNumberVerified,
    setFaceVerified,
    setSecondaryVerificationComplete,
    setVerificationNumbers,
    setCorrectNumber,
    setIsCameraActive,
    setFaceDetected,
    setFaceDetectionQuality,
    setLivenessDetected,
    setPreviousLandmarks,
    setLivenessChecks,
    setFaceDetectionMode,
    setFaceVerificationStage,
    setCaptureCount,
    setVerificationProgress,
    setModelsLoaded,
    setCameraPermission,
    setIsModelLoading,
    setModelLoadingError,
    setWebGLSupported,
    setIsLoading,
    setError,
    setAlertMessage,
    setAlertVariant,
    setShowSecurityInfo,
    setShowTroubleshooting,
    setShowAlternativeMethod,
    setVerificationMethod,
    setVerificationAttempts,
    setIsLockedOut,
    setLockoutTime,
    setSessionTimer,
    setSessionTimerActive,
    setDeviceFingerprint,
    setIpAddress,
    setGeoLocation,
    setFacialRecognitionFailed,
    setFacialAttempts,
    setShowAlternativeOption,
    formatTime,
    showAlert,
    handleFailedAttempt,
    generateVerificationNumbers,
    verifyNumber,
    stopCamera,
    handleSecondaryVerificationSuccess,
    handleSecondaryVerificationError,
    toggleSecurityInfo,
    toggleTroubleshooting,
    switchToAlternativeMethod,
    handleFacialRecognitionFailure,
    switchToAlternativeVerification,
  }

  return <VerificationContext.Provider value={value}>{children}</VerificationContext.Provider>
}

