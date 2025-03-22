"use client"

import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { CheckCircle, ChevronRight } from "lucide-react"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

// Button component
const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? React.Fragment : "button"

    let variantClasses = ""
    if (variant === "default") variantClasses = "bg-primary text-primary-foreground hover:bg-primary/90"
    else if (variant === "destructive")
      variantClasses = "bg-destructive text-destructive-foreground hover:bg-destructive/90"
    else if (variant === "outline")
      variantClasses = "border border-input bg-background hover:bg-accent hover:text-accent-foreground"
    else if (variant === "secondary") variantClasses = "bg-secondary text-secondary-foreground hover:bg-secondary/80"
    else if (variant === "ghost") variantClasses = "hover:bg-accent hover:text-accent-foreground"
    else if (variant === "link") variantClasses = "text-primary underline-offset-4 hover:underline"

    let sizeClasses = ""
    if (size === "default") sizeClasses = "h-10 px-4 py-2"
    else if (size === "sm") sizeClasses = "h-9 rounded-md px-3"
    else if (size === "lg") sizeClasses = "h-11 rounded-md px-8"
    else if (size === "icon") sizeClasses = "h-10 w-10"

    return (
      <Comp
        className={cn(
          "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          variantClasses,
          sizeClasses,
          className,
        )}
        ref={ref}
        {...props}
      />
    )
  },
)
Button.displayName = "Button"

export default function SuccessPage() {
  const navigate = useNavigate()
  const [matchedVoter, setMatchedVoter] = React.useState(null)
  const [countdown, setCountdown] = useState(5)

  useEffect(() => {
    // Check if user data exists in session storage
    const userData = sessionStorage.getItem("userData")
    const storedVoter = sessionStorage.getItem("matchedVoter")

    if (!userData) {
      // Redirect to home if no user data
      navigate("/")
      return
    }

    if (storedVoter) {
      const parsedVoter = JSON.parse(storedVoter)
      setMatchedVoter(parsedVoter)

      try {
        // Update userData with complete voter information including image/avatar
        // But optimize the avatar storage to prevent quota issues
        const updatedUserData = {
          name: parsedVoter.name,
          voterID: parsedVoter.id || parsedVoter.voterID,
          aadhaar: parsedVoter.aadhaar,
          phone: parsedVoter.phone,
          email: parsedVoter.email || "",
          dob: parsedVoter.dob,
          gender: parsedVoter.gender,
          address: parsedVoter.address,
          // Store avatar reference instead of the full data if it's a large string
          avatar:
            parsedVoter.avatar && parsedVoter.avatar.length > 1000
              ? `https://ui-avatars.com/api/?name=${encodeURIComponent(parsedVoter.name)}&background=random&color=fff&size=128`
              : parsedVoter.avatar,
          constituency: parsedVoter.constituency,
          pollingStation: parsedVoter.pollingStation,
          status: parsedVoter.status || "Verified",
        }

        // Store the complete user data in session storage
        sessionStorage.setItem("userData", JSON.stringify(updatedUserData))
      } catch (error) {
        console.error("Storage error:", error)

        // If storage fails, try a minimal version without the avatar
        if (error.name === "QuotaExceededError") {
          const minimalUserData = {
            name: parsedVoter.name,
            voterID: parsedVoter.id || parsedVoter.voterID,
            aadhaar: parsedVoter.aadhaar,
            phone: parsedVoter.phone,
            email: parsedVoter.email || "",
            dob: parsedVoter.dob,
            gender: parsedVoter.gender,
            address: parsedVoter.address,
            // Use a generated avatar instead
            avatar: `https://ui-avatars.com/api/?name=${encodeURIComponent(parsedVoter.name)}&background=random&color=fff&size=128`,
            constituency: parsedVoter.constituency,
            pollingStation: parsedVoter.pollingStation,
            status: parsedVoter.status || "Verified",
          }

          try {
            sessionStorage.setItem("userData", JSON.stringify(minimalUserData))
          } catch (secondError) {
            console.error("Still cannot store data:", secondError)
            // Last resort - store only essential data
            const essentialData = {
              name: parsedVoter.name,
              voterID: parsedVoter.id || parsedVoter.voterID,
              aadhaar: parsedVoter.aadhaar,
            }
            sessionStorage.setItem("userData", JSON.stringify(essentialData))
          }
        }
      }
    }

    // Set a timer to automatically redirect to dashboard after 5 seconds
    const redirectTimer = setTimeout(() => {
      navigate("/dashboard")
    }, 5000)

    // Clear the timer if the component unmounts
    return () => clearTimeout(redirectTimer)
  }, [navigate])

  // Add another useEffect for the countdown display
  useEffect(() => {
    const countdownInterval = setInterval(() => {
      setCountdown((prevCount) => {
        if (prevCount <= 1) {
          clearInterval(countdownInterval)
          return 0
        }
        return prevCount - 1
      })
    }, 1000)

    return () => clearInterval(countdownInterval)
  }, [])

  const goToDashboard = () => {
    navigate("/dashboard")
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-8">
        <div className="flex flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-24 w-24 items-center justify-center rounded-full bg-green-100">
            <CheckCircle className="h-12 w-12 text-green-600" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">
            <span className="bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
              Verification Successful!
            </span>
          </h1>
          <p className="mt-2 text-center text-sm text-gray-600">Your identity has been verified successfully.</p>
        </div>

        <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-md">
          <div className="border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50 px-6 py-4">
            <h2 className="text-lg font-medium text-gray-900">Verification Details</h2>
          </div>
          <div className="space-y-4 px-6 py-4">
            <div>
              <p className="text-sm font-medium text-gray-500">Status</p>
              <div className="mt-1 flex items-center">
                <span className="mr-2 flex h-6 w-6 items-center justify-center rounded-full bg-green-100">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </span>
                <p className="font-medium text-green-600">Verification Complete</p>
              </div>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-500">Voter Information</p>
              <div className="mt-1 rounded-md bg-gray-50 p-3">
                {matchedVoter ? (
                  <div className="space-y-2">
                    <p className="text-sm text-gray-700">
                      <span className="font-medium">Name:</span> {matchedVoter.name}
                    </p>
                    <p className="text-sm text-gray-700">
                      <span className="font-medium">Voter ID:</span> {matchedVoter.id || matchedVoter.voterID}
                    </p>
                    <p className="text-sm text-gray-700">
                      <span className="font-medium">Constituency:</span> {matchedVoter.constituency || "Not available"}
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-gray-700">
                    Your voter information has been verified. You can view your complete details in the dashboard.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
              <div className="flex items-start space-x-3">
                <div className="mt-0.5 text-blue-600">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                  </svg>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-blue-800">Important Information</h4>
                  <p className="mt-1 text-xs text-blue-700">
                    Your voter information is now accessible through your dashboard. Keep your voter ID and credentials
                    secure. Do not share them with anyone.
                  </p>
                </div>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-200 bg-gray-50 px-6 py-4">
            <Button
              onClick={goToDashboard}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white transition-all duration-300 hover:from-blue-700 hover:to-purple-700"
            >
              Go to Dashboard
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
            <p className="mt-2 text-center text-xs text-gray-500">Redirecting to dashboard in {countdown} seconds...</p>
          </div>
        </div>

        <div className="text-center text-xs text-gray-500">
          <p>
            Having trouble?{" "}
            <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">
              Contact support
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}

