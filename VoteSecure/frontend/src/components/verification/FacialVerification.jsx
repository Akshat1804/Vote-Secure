"use client"

import { useState, useRef, useEffect } from "react"
import { Camera, AlertCircle, UserCheck, HelpCircle } from "lucide-react"
import { Button } from "./UIComponents"
import { useVerification } from "../../context/VerificationContext"

const FacialVerification = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [photoTaken, setPhotoTaken] = useState(false)
  const [showPhotoPreview, setShowPhotoPreview] = useState(false)

  // Get verification context functions
  const { handleSecondaryVerificationSuccess, switchToAlternativeVerification } = useVerification()

  const startCamera = async () => {
    setError(null)
    setIsLoading(true)
    setPhotoTaken(false)
    setShowPhotoPreview(false)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.style.display = "block" // Ensure video is visible

        videoRef.current.onloadedmetadata = () => {
          videoRef.current
            .play()
            .then(() => {
              setIsStreaming(true)
              setIsLoading(false)
            })
            .catch((err) => {
              console.error("Error playing video:", err)
              setError("Error playing video. Please try again.")
              setIsLoading(false)
            })
        }
      }
    } catch (err) {
      console.error("Error accessing camera:", err)
      setError("Could not access camera. Please make sure you've granted permission.")
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }

  const takePhoto = () => {
    if (!videoRef.current || !isStreaming) return

    // Create canvas if it doesn't exist
    if (!canvasRef.current) {
      const canvas = document.createElement("canvas")
      canvasRef.current = canvas
    }

    const canvas = canvasRef.current
    const video = videoRef.current

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    const ctx = canvas.getContext("2d")
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // In a real app, you would now process this image for facial recognition
    console.log("Photo captured")

    // Show success state
    setPhotoTaken(true)
    setShowPhotoPreview(true)

    // In a real implementation, you would verify the photo here
    // For this demo, we'll simulate a successful verification after 2 seconds
    setTimeout(() => {
      handleSecondaryVerificationSuccess()
    }, 2000)
  }

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (isStreaming) {
        stopCamera()
      }
    }
  }, [isStreaming])

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <Camera className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Camera Access</h4>
            <p className="mt-1 text-xs text-blue-700">
              {!isStreaming
                ? "Click the button below to start the camera."
                : "Look directly at the camera and click 'Take Photo'."}
            </p>
          </div>
        </div>
      </div>

      <div className="relative mx-auto aspect-video w-full max-w-sm overflow-hidden rounded-lg border border-gray-200 bg-black">
        {!isStreaming && !error && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-white">
            Click the button below to start your camera
          </div>
        )}

        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent mb-2"></div>
            <p>Starting camera...</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-red-500 p-4 text-center">
            <AlertCircle className="mr-2 h-5 w-5" />
            {error}
          </div>
        )}

        {photoTaken && showPhotoPreview && (
          <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-white">
            <UserCheck className="h-16 w-16 text-green-500 mb-2" />
            <p className="text-xl font-bold">Photo captured!</p>
            <p className="text-sm">Verifying your identity...</p>
          </div>
        )}

        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="h-full w-full object-cover"
          style={{ display: isStreaming ? "block" : "none" }}
        />
      </div>

      <div className="flex flex-col space-y-3">
        {!isStreaming ? (
          <Button
            onClick={startCamera}
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
          >
            {isLoading ? "Starting Camera..." : "Start Camera"}
          </Button>
        ) : (
          <div className="flex flex-col space-y-3">
            <Button
              onClick={takePhoto}
              disabled={photoTaken}
              className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white hover:from-green-700 hover:to-teal-700"
            >
              {photoTaken ? "Photo Captured" : "Take Photo"}
            </Button>

            <Button
              onClick={stopCamera}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
            >
              Stop Camera
            </Button>
          </div>
        )}

        {/* Always show the alternative verification option */}
        <div className="mt-4 border-t pt-4">
          <div className="flex items-center justify-center mb-2 text-sm text-gray-500">
            <HelpCircle className="h-4 w-4 mr-1" />
            <span>Having trouble with camera verification?</span>
          </div>
          <Button
            onClick={switchToAlternativeVerification}
            className="w-full bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
          >
            Use Alternative Verification Method
          </Button>
        </div>
      </div>
    </div>
  )
}

export default FacialVerification

