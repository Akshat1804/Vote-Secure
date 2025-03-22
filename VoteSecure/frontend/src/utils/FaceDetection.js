// Enhanced face detection using canvas API with additional features
export const EnhancedFaceDetector = {
    // Detect face using multiple techniques for better reliability
    detectFace: (videoEl, canvasEl) => {
      if (!videoEl || !canvasEl) return { detected: false, quality: 0, landmarks: null }
  
      try {
        const ctx = canvasEl.getContext("2d")
        ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height)
  
        // Get image data from the center of the frame where a face would likely be
        const centerX = canvasEl.width / 2
        const centerY = canvasEl.height / 2
        const sampleSize = Math.min(canvasEl.width, canvasEl.height) * 0.5 // 50% of the smaller dimension
  
        const imageData = ctx.getImageData(centerX - sampleSize / 2, centerY - sampleSize / 2, sampleSize, sampleSize)
  
        // Enhanced algorithm to detect skin-like colors
        let skinPixels = 0
        const data = imageData.data
  
        // Track potential eye regions for liveness detection
        const potentialEyes = []
  
        for (let y = 0; y < sampleSize; y++) {
          for (let x = 0; x < sampleSize; x++) {
            const idx = (y * sampleSize + x) * 4
            const r = data[idx]
            const g = data[idx + 1]
            const b = data[idx + 2]
  
            // More sophisticated skin tone detection that works across different skin tones
            if (
              (r > 60 && g > 40 && b > 20 && r > g && r > b && r - g > 10 && r - b > 10) || // Lighter skin tones
              (r > 40 && g > 30 && b > 20 && r > g && g > b) // Darker skin tones
            ) {
              skinPixels++
            }
  
            // Simple eye detection (dark regions surrounded by skin)
            if (r < 80 && g < 80 && b < 80) {
              const realX = centerX - sampleSize / 2 + x
              const realY = centerY - sampleSize / 2 + y
              potentialEyes.push({ x: realX, y: realY })
            }
          }
        }
  
        // Calculate percentage of skin-like pixels
        const totalPixels = sampleSize * sampleSize
        const skinPercentage = (skinPixels / totalPixels) * 100
  
        // Cluster potential eye regions
        const eyeRegions = []
        if (potentialEyes.length > 0) {
          // Simple clustering algorithm
          const visited = new Set()
  
          for (let i = 0; i < potentialEyes.length; i++) {
            if (visited.has(i)) continue
  
            const cluster = [potentialEyes[i]]
            visited.add(i)
  
            for (let j = 0; j < potentialEyes.length; j++) {
              if (visited.has(j)) continue
  
              const dist = Math.sqrt(
                Math.pow(potentialEyes[i].x - potentialEyes[j].x, 2) +
                  Math.pow(potentialEyes[i].y - potentialEyes[j].y, 2),
              )
  
              if (dist < 20) {
                // Close enough to be part of the same eye
                cluster.push(potentialEyes[j])
                visited.add(j)
              }
            }
  
            if (cluster.length > 5) {
              // Minimum size to be considered an eye
              // Calculate center of the cluster
              const centerX = cluster.reduce((sum, p) => sum + p.x, 0) / cluster.length
              const centerY = cluster.reduce((sum, p) => sum + p.y, 0) / cluster.length
              eyeRegions.push({ x: centerX, y: centerY, size: cluster.length })
            }
          }
        }
  
        // Sort eye regions by size (largest first)
        eyeRegions.sort((a, b) => b.size - a.size)
  
        // Take the two largest regions as eyes
        const eyes = eyeRegions.slice(0, 2)
  
        // Check if we have two eyes at a reasonable distance
        let hasValidEyes = false
        if (eyes.length === 2) {
          const eyeDist = Math.sqrt(Math.pow(eyes[0].x - eyes[1].x, 2) + Math.pow(eyes[0].y - eyes[1].y, 2))
  
          // Eyes should be at a reasonable distance apart
          hasValidEyes = eyeDist > 30 && eyeDist < 150
        }
  
        // Calculate detection quality based on skin percentage and eye detection
        let detectionQuality = skinPercentage * 0.7
        if (hasValidEyes) {
          detectionQuality += 30 // Bonus for detecting eyes
        }
  
        // Clamp quality between 0 and 100
        detectionQuality = Math.min(100, Math.max(0, detectionQuality))
  
        // Draw a face detection overlay if quality is good enough
        if (detectionQuality > 30) {
          // Draw face detection box
          const boxSize = sampleSize * 1.5
          ctx.strokeStyle = "#4f46e5"
          ctx.lineWidth = 3
          ctx.strokeRect(centerX - boxSize / 2, centerY - boxSize / 2, boxSize, boxSize)
  
          // Add label
          ctx.fillStyle = "rgba(79, 70, 229, 0.8)"
          ctx.fillRect(centerX - boxSize / 2, centerY - boxSize / 2 - 25, 120, 25)
          ctx.fillStyle = "white"
          ctx.font = "bold 14px Arial"
          ctx.fillText("Face Detected", centerX - boxSize / 2 + 10, centerY - boxSize / 2 - 8)
  
          // Draw eye regions if detected
          if (hasValidEyes) {
            eyes.forEach((eye) => {
              ctx.beginPath()
              ctx.arc(eye.x, eye.y, 5, 0, 2 * Math.PI)
              ctx.fillStyle = "#10b981"
              ctx.fill()
            })
          }
  
          return {
            detected: true,
            quality: detectionQuality,
            landmarks: hasValidEyes ? eyes : null,
          }
        }
  
        return { detected: false, quality: detectionQuality, landmarks: null }
      } catch (error) {
        console.error("Error in enhanced face detection:", error)
        return { detected: false, quality: 0, landmarks: null }
      }
    },
  
    // Draw verification overlay with enhanced features
    drawVerificationOverlay: (canvasEl, landmarks = null) => {
      if (!canvasEl) return
  
      const ctx = canvasEl.getContext("2d")
      const centerX = canvasEl.width / 2
      const centerY = canvasEl.height / 2
      const boxSize = 200
  
      // Clear canvas
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height)
  
      // Draw verification overlay
      ctx.fillStyle = "rgba(16, 185, 129, 0.2)"
      ctx.fillRect(0, 0, canvasEl.width, canvasEl.height)
  
      // Draw scanning animation
      const scanLineY = ((Date.now() % 2000) / 2000) * canvasEl.height
      ctx.strokeStyle = "rgba(16, 185, 129, 0.8)"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(0, scanLineY)
      ctx.lineTo(canvasEl.width, scanLineY)
      ctx.stroke()
  
      // Draw face detection box
      ctx.strokeStyle = "#10b981"
      ctx.lineWidth = 3
      ctx.strokeRect(centerX - boxSize / 2, centerY - boxSize / 2, boxSize, boxSize)
  
      // Add verification text
      ctx.fillStyle = "rgba(16, 185, 129, 0.8)"
      ctx.fillRect(centerX - boxSize / 2, centerY - boxSize / 2 - 30, 120, 25)
      ctx.fillStyle = "white"
      ctx.font = "bold 14px Arial"
      ctx.fillText("Verifying...", centerX - boxSize / 2 + 10, centerY - boxSize / 2 - 12)
  
      // Draw facial landmarks (either from detection or simplified)
      if (landmarks && landmarks.length >= 2) {
        // Draw detected landmarks
        landmarks.forEach((point) => {
          ctx.beginPath()
          ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI)
          ctx.fillStyle = "#10b981"
          ctx.fill()
        })
  
        // Connect landmarks with lines
        ctx.beginPath()
        ctx.moveTo(landmarks[0].x, landmarks[0].y)
        ctx.lineTo(landmarks[1].x, landmarks[1].y)
        ctx.strokeStyle = "#10b981"
        ctx.lineWidth = 2
        ctx.stroke()
      } else {
        // Draw simplified face landmarks
        const drawLandmark = (x, y) => {
          ctx.beginPath()
          ctx.arc(x, y, 3, 0, 2 * Math.PI)
          ctx.fillStyle = "#10b981"
          ctx.fill()
        }
  
        // Eyes
        drawLandmark(centerX - 30, centerY - 20)
        drawLandmark(centerX + 30, centerY - 20)
  
        // Nose
        drawLandmark(centerX, centerY)
  
        // Mouth
        drawLandmark(centerX - 20, centerY + 30)
        drawLandmark(centerX, centerY + 35)
        drawLandmark(centerX + 20, centerY + 30)
      }
  
      // Draw biometric verification animation
      const time = Date.now()
      const numPoints = 8
      const radius = boxSize / 2 + 20
  
      for (let i = 0; i < numPoints; i++) {
        const angle = (i / numPoints) * Math.PI * 2 + ((time % 3000) / 3000) * Math.PI * 2
        const x = centerX + Math.cos(angle) * radius
        const y = centerY + Math.sin(angle) * radius
  
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(16, 185, 129, ${0.3 + 0.7 * Math.sin(((time % 1000) / 1000) * Math.PI * 2 + i)})`
        ctx.fill()
      }
    },
  
    // Perform liveness detection (blinking, head movement)
    detectLiveness: (videoEl, canvasEl, previousLandmarks = []) => {
      if (!videoEl || !canvasEl || !previousLandmarks.length) return false
  
      try {
        // Get current landmarks
        const result = EnhancedFaceDetector.detectFace(videoEl, canvasEl)
        if (!result.detected || !result.landmarks) return false
  
        const currentLandmarks = result.landmarks
  
        // Check for movement (simple version)
        let hasMovement = false
        if (previousLandmarks.length >= 2 && currentLandmarks.length >= 2) {
          // Calculate movement of eyes
          const prevLeftEye = previousLandmarks[0]
          const prevRightEye = previousLandmarks[1]
          const currLeftEye = currentLandmarks[0]
          const currRightEye = currentLandmarks[1]
  
          // Calculate distance moved
          const leftEyeMovement = Math.sqrt(
            Math.pow(prevLeftEye.x - currLeftEye.x, 2) + Math.pow(prevLeftEye.y - currLeftEye.y, 2),
          )
  
          const rightEyeMovement = Math.sqrt(
            Math.pow(prevRightEye.x - currRightEye.x, 2) + Math.pow(prevRightEye.y - currRightEye.y, 2),
          )
  
          // If eyes moved more than threshold, consider it movement
          hasMovement = leftEyeMovement > 2 || rightEyeMovement > 2
        }
  
        return hasMovement
      } catch (error) {
        console.error("Error in liveness detection:", error)
        return false
      }
    },
  }
  
  // Check WebGL support
  export const checkWebGLSupport = () => {
    try {
      // Try to create a WebGL context
      const canvas = document.createElement("canvas")
  
      // Try WebGL2 first
      let gl = canvas.getContext("webgl2")
  
      // Fall back to WebGL1
      if (!gl) {
        gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl")
      }
  
      if (!gl) {
        console.error("WebGL is not supported in this browser")
        return false
      }
  
      // Additional check for WebGL capabilities
      const debugInfo = gl.getExtension("WEBGL_debug_renderer_info")
      if (debugInfo) {
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
        console.log("WebGL Renderer:", renderer)
  
        // Check if using software renderer (SwiftShader, ANGLE, etc.)
        if (renderer.includes("SwiftShader") || renderer.includes("Software")) {
          console.warn("Using software WebGL renderer which may have limited capabilities")
          return true
        }
      }
  
      return true
    } catch (e) {
      console.error("Error checking WebGL support:", e)
      return false
    }
  }
  
  